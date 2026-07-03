import uuid
import streamlit as st
from agent.react_agent import MultiAgentWrapper
import traceback
import time

# ----------------------------
# 页面配置
# ----------------------------
if "agent" not in st.session_state:
    try:
        st.session_state["agent"] = MultiAgentWrapper()
    except Exception as e:
        st.error(f"❌ 初始化多智能体系统失败: {str(e)}")
        st.stop()

st.set_page_config(
    page_title="PaperMate - 论文阅读助手",
    page_icon="\U0001F4DA",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------
# 样式：极简 ChatGPT 风格 + 隐藏 Streamlit 默认元素
# ----------------------------
st.markdown("""
<style>
    /* 整体背景 */
    .stApp {
        background: #f7f7f8;
    }

    /* 隐藏 Streamlit 默认元素 */
    [data-testid="stToolbar"] {
        display: none !important;
    }

    [data-testid="stDecoration"] {
        display: none !important;
    }

    [data-testid="stHeader"] {
        display: none !important;
    }

    #MainMenu {
        visibility: hidden !important;
    }

    footer {
        visibility: hidden !important;
    }

    header {
        visibility: hidden !important;
    }

    /* 页面主体宽度 */
    .block-container {
        max-width: 980px;
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }

    html, body, [class*="css"] {
        color: #111827;
    }

    /* 侧边栏 */
    section[data-testid="stSidebar"] {
        background: #ececf1;
        border-right: 1px solid #d9d9e3;
        min-width: 290px !important;
        max-width: 290px !important;
    }

    .sidebar-title {
        font-size: 1.08rem;
        font-weight: 700;
        color: #111827;
        margin-bottom: 0.9rem;
    }

    /* 主标题区 */
    .main-title {
        font-size: 1.95rem;
        font-weight: 700;
        color: #111827;
        margin-bottom: 0.25rem;
        letter-spacing: -0.02em;
    }

    .sub-title {
        font-size: 0.98rem;
        color: #4b5563;
        line-height: 1.7;
        margin-bottom: 1.2rem;
    }

    /* 空状态 */
    .empty-wrap {
        text-align: center;
        padding-top: 9vh;
        color: #6b7280;
    }

    .empty-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: #111827;
        margin-bottom: 0.6rem;
    }

    .empty-desc {
        font-size: 1rem;
        color: #6b7280;
        line-height: 1.7;
    }

    /* 消息区 */
    div[data-testid="stChatMessage"] {
        background: transparent;
        border: none;
        padding-top: 0.3rem;
        padding-bottom: 0.3rem;
        margin-bottom: 0.35rem;
    }

    [data-testid="chat-avatar-icon-user"] svg,
    [data-testid="chat-avatar-icon-assistant"] svg {
        width: 1.15rem;
        height: 1.15rem;
    }

    /* 输入框区域 */
    .stChatInputContainer {
        background: #f7f7f8;
    }

    /* 历史记录说明 */
    .history-tip {
        font-size: 0.85rem;
        color: #6b7280;
        line-height: 1.6;
        margin-top: 0.8rem;
    }

    .quick-label {
        font-size: 0.92rem;
        font-weight: 600;
        color: #4b5563;
        margin-bottom: 0.75rem;
    }

    hr {
        border-color: #e5e7eb;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }

    /* 通用按钮 */
    div.stButton > button {
        width: 100%;
        border-radius: 14px;
        border: 1px solid #d1d5db;
        background: #ffffff;
        color: #111827;
        font-weight: 500;
        padding: 0.62rem 0.85rem;
        box-shadow: none;
        transition: all 0.2s ease;
    }

    div.stButton > button:hover {
        background: #f3f4f6;
        border-color: #c7cdd4;
        color: #111827;
        transform: translateY(-1px);
    }

    /* 左侧历史小框按钮更像卡片 */
    section[data-testid="stSidebar"] div.stButton > button {
        text-align: left !important;
        justify-content: flex-start !important;
        border-radius: 12px;
        border: 1px solid #d7d9df;
        background: #f8f8fb;
        color: #111827;
        padding: 0.72rem 0.85rem;
        margin-bottom: 0.35rem;
        font-size: 0.92rem;
        min-height: 48px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    section[data-testid="stSidebar"] div.stButton > button:hover {
        background: #ffffff;
        border-color: #bfc5cf;
        transform: translateX(2px);
    }

    /* 当前选中的历史记录卡片 */
    .active-history {
        background: #ffffff;
        border: 1px solid #bfc5cf;
        border-radius: 12px;
        padding: 0.72rem 0.85rem;
        margin-bottom: 0.35rem;
        color: #111827;
        font-size: 0.92rem;
        font-weight: 600;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }

    /* 新建对话按钮 */
    .new-chat-wrap {
        margin-bottom: 0.8rem;
    }

    /* 常用问题按钮更宽更长 */
    .quick-question-note {
        font-size: 0.88rem;
        color: #6b7280;
        margin-bottom: 0.8rem;
    }

    /* 加载动画 */
    .loading-dots::after {
        content: '...';
        animation: dots 1.5s steps(4, end) infinite;
    }

    @keyframes dots {
        0%, 20% { content: '.'; }
        40% { content: '..'; }
        60%, 100% { content: '...'; }
    }

    /* 状态标签 */
    .status-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 500;
        margin-left: 8px;
    }

    .status-success {
        background: #d1fae5;
        color: #065f46;
    }

    .status-error {
        background: #fee2e2;
        color: #991b1b;
    }

    .status-warning {
        background: #fef3c7;
        color: #92400e;
    }
</style>
""", unsafe_allow_html=True)


# ----------------------------
# 初始化
# ----------------------------
def init_session_state():
    """初始化会话状态"""
    if "agent" not in st.session_state:
        try:
            st.session_state["agent"] = MultiAgentWrapper()
        except Exception as e:
            st.error(f"❌ 初始化多智能体系统失败: {str(e)}")
            st.stop()

    if "conversations" not in st.session_state:
        first_id = str(uuid.uuid4())
        st.session_state["conversations"] = {
            first_id: {
                "title": "新对话",
                "messages": [],
                "created_at": time.time()
            }
        }
        st.session_state["current_conversation_id"] = first_id

    if "current_conversation_id" not in st.session_state:
        st.session_state["current_conversation_id"] = next(iter(st.session_state["conversations"]))

    if "pending_prompt" not in st.session_state:
        st.session_state["pending_prompt"] = None

    if "is_processing" not in st.session_state:
        st.session_state["is_processing"] = False


init_session_state()


def get_current_conversation():
    """获取当前会话"""
    conv_id = st.session_state["current_conversation_id"]
    return st.session_state["conversations"][conv_id]


def build_title_from_messages(messages: list[dict]) -> str:
    """从消息构建标题"""
    for msg in messages:
        if msg["role"] == "user" and msg["content"].strip():
            title = msg["content"].strip().replace("\n", " ")
            # 移除特殊字符和多余空格
            title = ' '.join(title.split())
            return title[:25] + ("..." if len(title) > 25 else "")
    return "新对话"


def create_new_conversation():
    """创建新对话"""
    new_id = str(uuid.uuid4())
    st.session_state["conversations"][new_id] = {
        "title": "新对话",
        "messages": [],
        "created_at": time.time()
    }
    st.session_state["current_conversation_id"] = new_id
    st.session_state["pending_prompt"] = None
    st.session_state["is_processing"] = False


def delete_conversation(conv_id: str):
    """删除对话"""
    if conv_id in st.session_state["conversations"]:
        # 如果删除的是当前对话，切换到第一个
        if conv_id == st.session_state["current_conversation_id"]:
            remaining = [cid for cid in st.session_state["conversations"] if cid != conv_id]
            if remaining:
                st.session_state["current_conversation_id"] = remaining[0]
            else:
                # 如果没有剩余对话，创建新对话
                create_new_conversation()
        del st.session_state["conversations"][conv_id]
        st.rerun()


# ----------------------------
# 左侧历史栏
# ----------------------------
with st.sidebar:
    st.markdown('<div class="sidebar-title">[PAPER] PaperMate</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([4, 1])
    with col1:
        if st.button("➕ 新建对话", key="new_chat_btn", use_container_width=True):
            create_new_conversation()
            st.rerun()

    with col2:
        # 显示当前状态
        if st.session_state["is_processing"]:
            st.markdown('<span class="status-badge status-warning">⏳ 处理中</span>', unsafe_allow_html=True)

    st.markdown("---")

    # 显示历史对话
    conversation_ids = list(st.session_state["conversations"].keys())

    if not conversation_ids:
        st.info("暂无对话，点击上方新建对话开始")
    else:
        # 按创建时间倒序显示
        sorted_conversations = sorted(
            conversation_ids,
            key=lambda x: st.session_state["conversations"][x].get("created_at", 0),
            reverse=True
        )

        for cid in sorted_conversations:
            conv = st.session_state["conversations"][cid]
            title = conv["title"]
            display_title = title if title.strip() else "新对话"

            # 显示消息数量
            msg_count = len(conv["messages"])
            title_with_count = f"{display_title} ({msg_count})"

            if cid == st.session_state["current_conversation_id"]:
                st.markdown(
                    f'<div class="active-history">{title_with_count}</div>',
                    unsafe_allow_html=True
                )
            else:
                col1, col2 = st.columns([5, 1])
                with col1:
                    if st.button(display_title, key=f"history_{cid}", use_container_width=True):
                        st.session_state["current_conversation_id"] = cid
                        st.session_state["pending_prompt"] = None
                        st.session_state["is_processing"] = False
                        st.rerun()
                with col2:
                    if st.button("✕", key=f"del_{cid}", help="删除此对话"):
                        delete_conversation(cid)

    st.markdown("---")

    # 显示系统信息
    with st.expander("[SYS] 系统信息"):
        st.markdown(f"""
        - **模型**: {st.session_state.get('model_name', 'Ollama Qwen 3')}
        - **会话数**: {len(conversation_ids)}
        - **状态**: {'运行中' if not st.session_state.get('is_processing') else '处理中'}
        """)

    st.markdown(
        '<div class="history-tip">点击历史对话可切换，点击 [X] 可删除。</div>',
        unsafe_allow_html=True
    )

# ----------------------------
# 右侧主聊天区
# ----------------------------
current_conv = get_current_conversation()
messages = current_conv["messages"]

# 显示标题
st.markdown(
    """
    <div class="main-title">[PAPER] 论文阅读助手 PaperMate</div>
    <div class="sub-title">
        基于 RAG 的本地论文知识库，支持论文内容检索、概念解释、方法对比与研究脉络梳理。
    </div>
    """,
    unsafe_allow_html=True
)

# 显示消息
if len(messages) == 0:
    # 空状态
    st.markdown("""
    <div class="empty-wrap">
        <div class="empty-title">今天想了解哪篇论文？</div>
        <div class="empty-desc">
            你可以直接提问，例如：<br>
            <strong>Transformer 的自注意力机制是如何工作的？</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="quick-label">研究问题示例</div>', unsafe_allow_html=True)
    st.markdown('<div class="quick-question-note">点击下面的问题可直接开始对话。</div>', unsafe_allow_html=True)

    # 常用问题网格
    col1, col2 = st.columns(2)

    quick_questions = [
        ("Transformer 的注意力机制原理是什么？", "Transformer 的注意力机制原理是什么？"),
        ("BERT 和 GPT 的预训练方法有什么不同？", "BERT 和 GPT 的预训练方法有什么不同？"),
        ("Word2Vec 和 GloVe 的词向量方法有什么区别？", "Word2Vec 和 GloVe 的词向量方法有什么区别？"),
        ("Seq2Seq 模型中的注意力机制是如何实现的？", "Seq2Seq 模型中的注意力机制是如何实现的？"),
    ]

    for i, (label, question) in enumerate(quick_questions):
        if i % 2 == 0:
            with col1:
                if st.button(label, key=f"quick_q_{i}", use_container_width=True):
                    st.session_state["pending_prompt"] = question
                    st.rerun()
        else:
            with col2:
                if st.button(label, key=f"quick_q_{i}", use_container_width=True):
                    st.session_state["pending_prompt"] = question
                    st.rerun()

    st.markdown("---")
else:
    # 显示历史消息
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# ----------------------------
# 输入处理
# ----------------------------
user_input = st.chat_input("请输入你的研究问题...", disabled=st.session_state["is_processing"])

# 处理待处理的提示
if st.session_state["pending_prompt"]:
    prompt = st.session_state["pending_prompt"]
    st.session_state["pending_prompt"] = None
elif user_input:
    prompt = user_input
else:
    prompt = None

# ----------------------------
# 流式输出
# ----------------------------
if prompt and not st.session_state["is_processing"]:
    # 设置处理状态
    st.session_state["is_processing"] = True

    # 获取当前会话
    current_conv = get_current_conversation()

    # 添加用户消息
    current_conv["messages"].append({"role": "user", "content": prompt})
    current_conv["title"] = build_title_from_messages(current_conv["messages"])

    # 显示用户消息
    with st.chat_message("user"):
        st.markdown(prompt)

    # 显示助手消息
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""
        error_occurred = False

        try:
            # 获取 Agent 并执行
            agent = st.session_state["agent"]
            stream = agent.execute_stream(prompt)

            # 流式输出
            for chunk in stream:
                if chunk:
                    full_response += chunk
                    placeholder.markdown(full_response + "▌")

            # 显示完整响应
            if full_response:
                placeholder.markdown(full_response)
            else:
                full_response = "（无响应内容）"
                placeholder.info(full_response)

        except Exception as e:
            error_occurred = True
            error_msg = traceback.format_exc()
            print(f"[ERROR] 错误详情:\n{error_msg}")

            full_response = f"❌ 系统运行出错：{type(e).__name__}\n\n错误信息：{str(e)}"
            placeholder.error(full_response)

        # 保存助手消息
        current_conv["messages"].append({"role": "assistant", "content": full_response})
        current_conv["title"] = build_title_from_messages(current_conv["messages"])

    # 重置处理状态
    st.session_state["is_processing"] = False

    # 刷新页面
    st.rerun()

# ----------------------------
# 底部提示
# ----------------------------
st.markdown("""
<div style="
    position: fixed;
    bottom: 10px;
    left: 50%;
    transform: translateX(-50%);
    font-size: 0.75rem;
    color: #9ca3af;
    text-align: center;
    z-index: 999;
    background: rgba(247, 247, 248, 0.9);
    padding: 4px 16px;
    border-radius: 20px;
    backdrop-filter: blur(10px);
">
    PaperMate v1.0 · 基于 LangChain RAG + 本地论文知识库
</div>
""", unsafe_allow_html=True)