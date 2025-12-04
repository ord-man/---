# -*- coding: utf-8 -*-

"""
作者：王艺
学校：sau
说明：集成Neo4j节点关系可视化功能，基于本地Transformers模型的交互平台
"""

import os as _os

_os.environ.setdefault("PYTHONUTF8", "1")
_os.environ.setdefault("PYTHONIOENCODING", "utf-8")
del _os
from flask import Flask, render_template, request, jsonify
import base64
import io
import cv2
import numpy as np
import speech_recognition as sr
from datetime import datetime
import os
import tempfile

# ==== ASR 可选配置（none / vosk）====
ASR_BACKEND = "vosk"
# ASR_BACKEND = os.environ.get("ASR_BACKEND", "none")  # 默认关闭：'none'；要本地中文识别，设 'vosk'
VOSK_MODEL_PATH = os.environ.get("VOSK_MODEL_PATH", r"vosk-model-small-cn-0.22")  # 改成你放中文模型的路径
_vosk_available = False
vosk_model = None
if ASR_BACKEND.lower() == "vosk":
    try:
        from vosk import Model as VoskModel  # 延迟导入

        if not os.path.isdir(VOSK_MODEL_PATH):
            raise RuntimeError(f"VOSK_MODEL_PATH 不存在: {VOSK_MODEL_PATH}")
        vosk_model = VoskModel(VOSK_MODEL_PATH)
        _vosk_available = True
    except Exception as e:
        print(f"[WARN] Vosk 初始化失败：{e}")
        _vosk_available = False
        vosk_model = None

#  基本可调参数（只保留本地模型相关）
# 直接写死你的本地模型目录，或用环境变量覆盖：
HF_MODEL_PATH = os.environ.get("HF_MODEL_PATH", r"E:\wym\model\deepseek-ai\sau-model-manual2")
HF_DEVICE = os.environ.get("HF_DEVICE", "auto")  # "auto" / "cuda" / "cpu"
HF_MAX_NEW_TOKENS = int(os.environ.get("HF_MAX_NEW_TOKENS", "512"))
HF_TEMPERATURE = float(os.environ.get("HF_TEMPERATURE", "0.7"))
HF_TOP_P = float(os.environ.get("HF_TOP_P", "0.95"))

# 仅需这些依赖
import ffmpeg
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Neo4j 配置与连接
from neo4j import GraphDatabase  # 导入Neo4j驱动

# Neo4j配置（替换为你的实际信息）
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "12345678"  # 默认为neo4j，首次登录需修改
neo4j_connected = False
neo4j_handler = None


# 初始化Neo4j连接
class Neo4jHandler:
    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    def query(self, cypher):
        with self.driver.session() as session:
            return session.run(cypher).data()


# 尝试连接Neo4j
try:
    neo4j_handler = Neo4jHandler()
    # 测试连接
    neo4j_handler.query("MATCH (n) RETURN count(n) LIMIT 1")
    neo4j_connected = True
    print("[Neo4j] 连接成功")
except Exception as e:
    print(f"[Neo4j] 连接失败：{e}")
    neo4j_connected = False

# 确保上传目录存在
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
import tempfile, contextlib, wave, json  # 后续用到的库


def _mktemp_path(suffix=".wav"):
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)  # 立刻关闭，避免 Windows 锁
    return path


app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# 本地 Transformers 调用
class LocalLLM:
    """
    直接加载 LLaMA-Factory 导出的本地合并模型进行推理。
    """

    def __init__(self):
        print(f"[LLM] 正在加载本地模型：{HF_MODEL_PATH}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            HF_MODEL_PATH, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            HF_MODEL_PATH,
            trust_remote_code=True,
            device_map=HF_DEVICE,  # "auto"/"cuda"/"cpu"
            torch_dtype="auto"  # 自动选择（有显卡一般是 bf16/fp16）
        )
        print("[LLM] 模型加载完成。")

        # 尝试获取 chat_template（很多经 LLaMA-Factory 导出的模型都带）
        self.chat_template = getattr(self.tokenizer, "chat_template", None)

    def _build_prompt(self, user_text: str, system_text: str):
        """
        优先使用 tokenizer 自带 chat_template；否则退回到通用指令格式。
        """
        if self.chat_template:
            messages = [
                {"role": "system", "content": system_text},
                {"role": "user", "content": user_text},
            ]
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception as e:
                print("[LLM] chat_template 应用失败，使用 fallback 提示词。", e)

        # fallback：常见 SFT 模板（DeepSeek/Qwen 系列多能识别）
        return f"<|system|>\n{system_text}\n<|user|>\n{user_text}\n<|assistant|>\n"

    @torch.inference_mode()
    def chat(self, user_text: str, system_text: str = "你是一个有帮助的中文助理。"):
        prompt = self._build_prompt(user_text, system_text)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=HF_MAX_NEW_TOKENS,
            do_sample=True,
            temperature=HF_TEMPERATURE,
            top_p=HF_TOP_P,
            eos_token_id=self.tokenizer.eos_token_id
        )

        full_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # 尝试从最后一个 assistant 段落截取
        if "<|assistant|>" in full_text:
            return full_text.split("<|assistant|>")[-1].strip()
        # 如果模型自带 chat_template，有时需要从 prompt 之后截取新增部分
        gen_text = full_text[len(self.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)):]
        return gen_text.strip() if gen_text.strip() else full_text.strip()

    def process_text(self, text: str) -> str:
        return self.chat(text)

    def process_image(self, image):
        # 这里仍是占位：如果以后换成多模态模型，在此接入（例如 Qwen2-VL、MiniCPM-V 等）
        h, w, _ = image.shape
        return f"（占位）当前未接入视觉推理。图像尺寸：{w}x{h}"


# 初始化本地模型与语音识别器
ai_model = LocalLLM()
recognizer = sr.Recognizer()


# Neo4j图谱可视化辅助函数
def generate_node_label(labels, props, keys):
    """生成节点显示标签"""
    if not labels:
        return "Unknown Node"

    primary_label = labels[0]

    # 常见的显示属性优先级
    display_props = ['name', 'title', 'label', 'id', 'username', 'email', 'company_name', 'project_name']

    display_value = None
    for prop_name in display_props:
        if prop_name in props and props[prop_name]:
            display_value = str(props[prop_name])
            break

    # 如果没找到常见属性，使用第一个可用属性
    if not display_value and props:
        first_key = list(props.keys())[0]
        display_value = str(props[first_key])

    if display_value:
        # 限制显示长度
        if len(display_value) > 15:
            display_value = display_value[:12] + "..."
        return f"{primary_label}\n{display_value}"
    else:
        return primary_label


def generate_node_tooltip(labels, props):
    """生成节点悬停提示"""
    tooltip_parts = []

    if labels:
        tooltip_parts.append(f"标签: {', '.join(labels)}")

    if props:
        tooltip_parts.append("属性:")
        for key, value in list(props.items())[:5]:  # 最多显示5个属性
            if value is not None:
                tooltip_parts.append(f"  {key}: {str(value)[:50]}")

    return '\n'.join(tooltip_parts)


#
@app.route('/')
def index():
    return render_template('index.html')  # 前端页面
# 【新增】添加管理员页面的路由
@app.route('/admin.html')
def admin():
    return render_template('admin.html')

@app.route('/api/process-text', methods=['POST'])
def process_text():
    data = request.json or {}
    text = data.get('text', '')
    if not text:
        return jsonify({'error': '请输入文本'}), 400
    try:
        result = ai_model.process_text(text)
        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/process-voice', methods=['POST'])
def process_voice():
    in_path = norm_path = None
    try:
        if 'audio' not in request.files:
            return jsonify({'error': '小苏未接收到音频文件 (form-data 字段名应为 audio)',
                            'transcript': '', 'result': '', 'note': ''}), 400

        wav_file = request.files['audio']
        # 1) 只生成路径，不持有文件句柄
        in_path = _mktemp_path('.wav')
        wav_file.save(in_path)  # werkzeug 会自行打开/写入/关闭

        # 2) 规范化到 16k/mono（若 ffmpeg 可用）
        wav_for_asr = in_path
        # 2) 规范化到 16k/mono
        wav_for_asr = in_path
        try:
            norm_path = _mktemp_path('.wav')
            print(f"[DEBUG] 正在尝试转换音频: {in_path} -> {norm_path}")

            # 修改点：捕获标准输出和错误输出，不要 quiet=True
            (
                ffmpeg
                .input(in_path)
                .output(norm_path, format='wav', acodec='pcm_s16le', ac=1, ar='16000')
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
            wav_for_asr = norm_path
            print("[DEBUG] FFmpeg 转换成功！")

        except ffmpeg.Error as e:
            #  这里会打印出真正的错误原因
            print("\n" + "!" * 50)
            print("[FFmpeg 致命错误] 转换失败！原因如下：")
            # 打印 stderr，这是 FFmpeg 告诉你的具体错误
            print(e.stderr.decode('utf-8', errors='ignore'))
            print("!" * 50 + "\n")
            # 既然转换失败，不要继续往下走了，直接让前端报错，方便调试
            return jsonify(
                {'error': '服务器音频转换失败，请检查后端日志', 'transcript': '', 'result': '', 'note': ''}), 500

        except Exception as e:
            print(f"[未知错误] FFmpeg 调用异常: {e}")
            # 同样直接返回错误，不要硬传 WebM 给 wave 库
            return jsonify({'error': f'音频转换异常: {str(e)}', 'transcript': '', 'result': '', 'note': ''}), 500

        # 3) 未启用 ASR：仅回执
        if ASR_BACKEND.lower() == 'none':
            return jsonify({'error': '', 'transcript': '', 'result': '',
                            'note': '已收到 WAV。当前未启用本地 ASR，请使用文本输入框与模型对话，或将 ASR_BACKEND=vosk 启用离线中文识别。'})

        # 4) 启用 Vosk：纯本地中文识别
        if ASR_BACKEND.lower() == 'vosk':
            if not _vosk_available or vosk_model is None:
                return jsonify(
                    {'error': 'Vosk 未正确初始化，请确认 pip install vosk 且 VOSK_MODEL_PATH 指向有效中文模型目录',
                     'transcript': '', 'result': '', 'note': ''}), 500

            from vosk import KaldiRecognizer  # 延迟导入
            transcript_parts = []
            with contextlib.closing(wave.open(wav_for_asr, 'rb')) as wf:
                rec = KaldiRecognizer(vosk_model, wf.getframerate())
                rec.SetWords(False)
                while True:
                    data = wf.readframes(4000)
                    if not data:
                        break
                    if rec.AcceptWaveform(data):
                        r = json.loads(rec.Result())
                        if r.get('text'):
                            transcript_parts.append(r['text'])
                r = json.loads(rec.FinalResult())
                if r.get('text'):
                    transcript_parts.append(r['text'])

            text = ' '.join(transcript_parts).strip()
            if not text:
                return jsonify({'error': '小苏 未识别到有效中文，请重试或检查录音质量',
                                'transcript': '', 'result': '', 'note': ''}), 400

            result = ai_model.process_text(text)
            return jsonify({'error': '', 'transcript': text, 'result': result, 'note': ''})

        return jsonify({'error': f'未知 ASR_BACKEND: {ASR_BACKEND}',
                        'transcript': '', 'result': '', 'note': ''}), 400

    except Exception as e:
        import traceback
        print("!!! 语音处理崩溃，详细报错如下 !!!")
        traceback.print_exc()  # 这会将具体错误打印在控制台
        return jsonify({'error': f"处理失败: {str(e)}", 'transcript': '', 'result': '', 'note': ''}), 500

    finally:
        # 5) 统一清理（即使 ffmpeg/ASR 报错也不留垃圾文件）
        for p in (in_path, norm_path):
            if p and os.path.exists(p):
                with contextlib.suppress(Exception):
                    os.remove(p)


@app.route('/api/process-image', methods=['POST'])
def process_image():
    try:
        if 'image' not in request.files:
            return jsonify({'error': '未上传图像'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': '未选择图像'}), 400

        image_stream = io.BytesIO(file.read())
        image_array = np.frombuffer(image_stream.getvalue(), dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        result = ai_model.process_image(image)

        _, buffer = cv2.imencode('.jpg', image_rgb)
        image_base64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({'result': result, 'image': image_base64})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/neo4j-visualize', methods=['POST'])
def neo4j_visualize():
    """查询Neo4j节点和关系，返回可视化数据"""
    if not neo4j_connected or not neo4j_handler:
        return jsonify({'error': 'Neo4j未连接，请检查配置和数据库状态'}), 500

    data = request.json or {}
    label = data.get('label', '')

    try:
        # 改进的Cypher查询
        if label:
            cypher = f"""
            MATCH (n:{label})
            OPTIONAL MATCH (n)-[r]->(m)
            RETURN 
                n, 
                labels(n) as n_labels, 
                keys(n) as n_keys,
                r, 
                type(r) as r_type, 
                m, 
                labels(m) as m_labels,
                keys(m) as m_keys
            LIMIT 100
            """
        else:
            cypher = """
            MATCH (n)
            OPTIONAL MATCH (n)-[r]->(m)
            RETURN 
                n, 
                labels(n) as n_labels, 
                keys(n) as n_keys,
                r, 
                type(r) as r_type, 
                m, 
                labels(m) as m_labels,
                keys(m) as m_keys
            LIMIT 100
            """

        result = neo4j_handler.query(cypher)
        print(f"[DEBUG] Neo4j查询返回 {len(result)} 条记录")

        # 打印前几条数据用于调试
        for i, item in enumerate(result[:3]):
            print(f"[DEBUG] 记录 {i + 1}:")
            n_dict = dict(item['n']) if item['n'] else {}
            print(f"  节点n: {n_dict}, 标签: {item['n_labels']}, 属性键: {item['n_keys']}")
            if item.get('m'):
                m_dict = dict(item['m']) if item['m'] else {}
                print(f"  节点m: {m_dict}, 标签: {item['m_labels']}, 属性键: {item['m_keys']}")
            if item.get('r'):
                print(f"  关系: {item['r_type']}")
            print("---")

    except Exception as e:
        print(f"[ERROR] Neo4j查询失败: {e}")
        return jsonify({'error': f'查询失败：{str(e)}'}), 500

    # 转换为可视化格式
    nodes = []
    edges = []
    node_ids = set()

    for item in result:
        # 处理主节点
        n = item['n']
        if n:
            # 使用更安全的ID生成方式
            try:
                n_id = str(n.element_id) if hasattr(n, 'element_id') else str(n.id)
            except:
                n_id = str(hash(str(dict(n))))

            if n_id not in node_ids:
                # 获取节点信息
                n_props = dict(n.items()) if n else {}
                n_labels = item.get('n_labels', [])
                n_keys = item.get('n_keys', [])

                # 改进的节点标签生成逻辑
                node_label = generate_node_label(n_labels, n_props, n_keys)

                nodes.append({
                    'id': n_id,
                    'label': node_label,
                    'group': n_labels[0] if n_labels else 'Unknown',
                    'title': generate_node_tooltip(n_labels, n_props)
                })
                node_ids.add(n_id)

        # 处理关系和目标节点
        m = item.get('m')
        r = item.get('r')

        if m and r and n:  # 确保所有必要元素都存在
            try:
                m_id = str(m.element_id) if hasattr(m, 'element_id') else str(m.id)
            except:
                m_id = str(hash(str(dict(m))))

            if m_id not in node_ids:
                m_props = dict(m.items()) if m else {}
                m_labels = item.get('m_labels', [])
                m_keys = item.get('m_keys', [])

                node_label = generate_node_label(m_labels, m_props, m_keys)

                nodes.append({
                    'id': m_id,
                    'label': node_label,
                    'group': m_labels[0] if m_labels else 'Unknown',
                    'title': generate_node_tooltip(m_labels, m_props)
                })
                node_ids.add(m_id)

            # 添加关系
            try:
                n_id = str(n.element_id) if hasattr(n, 'element_id') else str(n.id)
            except:
                n_id = str(hash(str(dict(n))))

            edges.append({
                'from': n_id,
                'to': m_id,
                'label': item.get('r_type', 'RELATED'),
                'title': f"关系类型: {item.get('r_type', 'RELATED')}"
            })

    print(f"[DEBUG] 生成节点数: {len(nodes)}, 关系数: {len(edges)}")

    # 如果没有数据，返回提示信息
    if len(nodes) == 0:
        return jsonify({
            'error': '数据库中没有找到数据。请确认：1) Neo4j数据库已启动 2) 数据库中包含节点和关系 3) 连接配置正确'
        }), 404

    return jsonify({
        'nodes': nodes,
        'edges': edges,
        'debug': {
            'total_records': len(result),
            'nodes_count': len(nodes),
            'edges_count': len(edges),
            'has_relationships': len(edges) > 0
        }
    })


if __name__ == '__main__':
    print(f"[启动] Neo4j连接状态: {'已连接' if neo4j_connected else ' 未连接'}")
    print(f"[启动] 语音识别: {' Vosk已启用' if _vosk_available else ' 已禁用'}")
    print(f"[启动] 本地模型: {HF_MODEL_PATH}")

    # 生产环境推荐用 gunicorn/uvicorn；这里保留原地跑法
    app.run(debug=True, host='0.0.0.0', port=5000)




