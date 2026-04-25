# Made by Adele.
import re
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio

# 1. Page Configuration
st.set_page_config(page_title="🌐 Internet Slang Dashboard", layout="wide")

with st.expander("📝 View Scoring Standard", expanded=False):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Awareness Score (Hearing):**")
        st.write("- 2: Frequently heard / Highly aware")
        st.write("- 1: Vaguely familiar / Heard of")
        st.write("- 0: Never heard of")
    with c2:
        st.markdown("**Usage Score:**")
        st.write("- 2: Frequently use in daily life")
        st.write("- 1: Occasionally use")
        st.write("- 0: Never use")
        
# Geographic Coordinates Mapping
GEO_MAP = {
    '澳门': [22.1987, 113.5439], '广东': [23.1291, 113.2644], '江西': [28.6761, 115.8921],
    '北京': [39.9042, 116.4074], '河北': [38.0428, 114.5149], '四川': [30.5728, 104.0668],
    '重庆': [29.5630, 106.5516], '浙江': [30.2741, 120.1551], '上海': [31.2304, 121.4737],
    '山东': [36.6512, 117.0483], '山西': [37.8706, 112.5489], '黑龙江': [45.7569, 126.6424],
    '湖北': [30.5928, 114.3055], '安徽': [31.8612, 117.2830], '辽宁': [41.8057, 123.4315],
    '广西': [22.8170, 108.3200], '江苏': [32.0603, 118.7969], '天津': [39.0841, 117.2008]
}

# Slang Content Mapping
SLANG_CONTENT = [
    "wc / 操 / 我靠",
    "老己 / 爱你老己",
    "我的身材很曼妙",
    "你 / 我已急哭",
    "洗衣粉儿",
    "心理委员我不得劲儿",
    "我不行了"
]

# ====================== 全局 Plotly 高清下载配置 ======================
HIGH_RES_CONFIG = {
    "toImageButtonOptions": {
        "format": "png",
        "height": 1000,
        "width": 1600,
        "scale": 3   # 3倍高清，论文级清晰度
    }
}
# =====================================================================

@st.cache_data
def load_and_fully_clean_data(file_path):
    df = pd.read_csv(file_path, encoding='utf-8-sig')
    df.columns = df.columns.str.strip()
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

    # 1. 修正列名映射逻辑 (关键修复点：移出了子函数内部)
    temp_cols = list(df.columns)
    for i in range(7):
        # 根据你的 CSV 结构，热梗分数列通常从第 11 列开始
        if len(temp_cols) > 11 + i:
            temp_cols[11 + i] = f"Score_Aw_{i}"
        if len(temp_cols) > 19 + i:
            temp_cols[19 + i] = f"Score_Us_{i}"
    df.columns = temp_cols

    rename_dict = {
        '序号': 'Order', '来自IP': 'IP', '1、您的性别': 'Gender',
        '2、您的年龄段': 'Age', '3、您的常住地': 'Residence',
        '4、您是否为澳门大学在读学生': 'UM Student',
        '5、您日常刷短视频的频率': 'Frequency',
        '7、您获取网络热梗的主要渠道有哪些': 'Acquisition Channel',
        '9、您通常在哪些场景下会使用网络热梗': 'Using Scene',
        '10、您认为网络热梗对您的日常交流有何影响': 'Influence'
    }
    df.rename(columns=rename_dict, inplace=True)

    # 2. Gender Cleaning
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].replace({'女': 'Female', '男': 'Male'})
    
    # 3. Age Cleaning
    if 'Age' in df.columns:
        df['Age'] = df['Age'].replace({'18岁以下': 'Under 18', '18～30岁': '18-30', '30岁以上': 'Over 30'})
        df['Age'] = pd.Categorical(df['Age'], categories=['Under 18', '18-30', 'Over 30'], ordered=True)
    
    # 4. Frequency Cleaning
    if 'Frequency' in df.columns:
        freq_map = {'经常': 'Frequently', '有时': 'Sometimes', '几乎不': 'Almost Never'}
        df['Frequency'] = df['Frequency'].replace(freq_map)
        df['Frequency'] = pd.Categorical(df['Frequency'], categories=['Almost Never', 'Sometimes', 'Frequently'], ordered=True)

    # 5. Impact
    impact_map = {
        '有积极影响': 'Positive Impact',
        '有负面影响': 'Negative Impact',
        '基本无影响': 'No Significant Impact',
        '无影响': 'No Significant Impact'
    }
    if 'Influence' in df.columns:
        df['Influence'] = df['Influence'].replace(impact_map)
    if 'UM Student' in df.columns:
        df['UM Student'] = df['UM Student'].replace({'是': 'UM Student', '否': 'Non-UM (General Public)'})

    # 5. Scoring Conversion
    cleanup_map = {'经常刷/听到': 2, '有印象': 1, '没听过': 0, '经常会用': 2, '有时会用': 1, '从来不用': 0}

    def safe_convert(series):
        return pd.to_numeric(series.replace(cleanup_map), errors='coerce').fillna(0).astype(float)

    h_cols = [f"Score_Aw_{i}" for i in range(7)]
    u_cols = [f"Score_Us_{i}" for i in range(7)]

    for col in h_cols + u_cols:
        if col in df.columns:
            df[col] = safe_convert(df[col])

    df['Hearing Score'] = df[h_cols].sum(axis=1)
    df['Using Score'] = df[u_cols].sum(axis=1)
    df['Total Score'] = df['Hearing Score'] + df['Using Score']

    # 6. Location Extraction
    def extract_loc(val):
        match = re.search(r'\((.*?)\)', str(val))
        return match.group(1).split('-')[0] if match else "Unknown"

    df['Location Name'] = df['IP'].apply(extract_loc)

    def get_lat_lon_with_jitter(loc):
        base = GEO_MAP.get(loc, [None, None])
        if base[0] is not None:
            return base[0] + np.random.uniform(-0.15, 0.15), base[1] + np.random.uniform(-0.15, 0.15)
        return None, None

    coords = df['Location Name'].apply(get_lat_lon_with_jitter)
    df['lat'] = [c[0] for c in coords]
    df['lon'] = [c[1] for c in coords]

    return df

def process_multi_choice_with_percentages(series, translation_map):
    all_translated = []
    raw_others = []
    for val in series.dropna():
        items = str(val).split('┋')
        for i in items:
            name = i.strip()
            if name in translation_map:
                all_translated.append(translation_map[name])
            else:
                all_translated.append('Others')
                if name not in ['其他', 'Others', '']:
                    raw_others.append(name)
    
    # Calculate Counts
    counts = pd.Series(all_translated).value_counts().reset_index()
    counts.columns = ['Item', 'Count']
    
    # Calculate Percentage
    total = counts['Count'].sum()
    counts['Percentage'] = (counts['Count'] / total * 100).round(1)
    # Create a label string for the chart: "Count (Percentage%)"
    counts['Label'] = counts.apply(lambda x: f"{x['Count']} ({x['Percentage']}%)", axis=1)
    
    return counts, list(set(raw_others))
    
# 定义翻译字典 (放在主程序全局位置)
CHANNEL_MAP = {
    '短视频平台': 'Short Video Platforms',
    '社交软件': 'Social Apps',
    '新闻资讯': 'News & Media',
    '日常交流': 'Daily Conversations',
    '综艺节目': 'Variety Shows',
    '影视作品': 'Movies & TV',
    '其他': 'Others'
}

SCENE_MAP = {
    '线下闲聊': 'Offline Chatting',
    '线上私聊（微信、QQ等）': 'Online Private Chat',
    '社交媒体互动（发帖、评论、转发等）': 'Social Media Interaction',
    '工作沟通': 'Formal Work/Study',
    '游戏交流': 'Gaming Chat',
    '特殊场合（体育赛事、演唱会等）': 'Gaming Chat',
    '其他': 'Others'
}

# --- Data Loading ---
try:
    df = load_and_fully_clean_data('internet_slang.csv')
except Exception as e:
    st.error(f"Error Loading Data: {e}")
    st.stop()

# --- Sidebar Filters ---
st.sidebar.header("Filters")
gen_options = df['Gender'].unique().tolist()
age_options = df['Age'].unique().tolist()
gen_list = st.sidebar.multiselect("Gender", gen_options, default=gen_options)
age_list = st.sidebar.multiselect("Age Group", age_options, default=age_options)

f_df = df[(df['Gender'].isin(gen_list)) & (df['Age'].isin(age_list))]

# --- UI Layout ---
st.title("🌐 Internet Slang Analytics Dashboard")

# 1. Key Metrics & Pie Charts
st.divider()
kpi_col, pie_col1, pie_col2 = st.columns([2, 1, 1])

with kpi_col:
    st.markdown("#### 📈 Key Metrics")
    m1, m2 = st.columns(2)
    m1.metric("Avg Awareness", round(f_df['Hearing Score'].mean(), 2) if not f_df.empty else 0)
    m2.metric("Avg Usage", round(f_df['Using Score'].mean(), 2) if not f_df.empty else 0)

    m3, m4 = st.columns(2)
    m3.metric("Avg Total Score", round(f_df['Total Score'].mean(), 2) if not f_df.empty else 0)
    m4.metric("Samples Count", len(f_df))

with pie_col1:
    if not f_df.empty:
        gender_counts = f_df['Gender'].value_counts().reset_index()
        gender_counts.columns = ['Gender', 'Count']
        fig_gen = px.pie(gender_counts, values='Count', names='Gender', hole=0.5,
                         title="Gender", color_discrete_sequence=px.colors.qualitative.Pastel)
        fig_gen.update_layout(height=250, margin=dict(t=40, b=0, l=0, r=0), showlegend=False)
        fig_gen.config.update(HIGH_RES_CONFIG)
        st.plotly_chart(fig_gen, use_container_width=True)

with pie_col2:
    if not f_df.empty and 'UM Student' in f_df.columns:
        um_counts = f_df['UM Student'].value_counts().reset_index()
        um_counts.columns = ['Status', 'Count']
        fig_um = px.pie(um_counts, values='Count', names='Status', hole=0.5,
                        title="Identity", color_discrete_sequence=px.colors.qualitative.Set3)
        fig_um.update_layout(height=250, margin=dict(t=40, b=0, l=0, r=0), showlegend=False)
        fig_um.config.update(HIGH_RES_CONFIG)
        st.plotly_chart(fig_um, use_container_width=True)

# 2. Geographic Distribution
st.divider()
st.subheader("📍 Geographic Distribution")
map_data = f_df.dropna(subset=['lat'])
if not map_data.empty:
    fig_map = px.scatter_mapbox(map_data, lat="lat", lon="lon", color="Gender", hover_name="Location Name",
                                zoom=3, height=450, mapbox_style="open-street-map")
    fig_map.config.update(HIGH_RES_CONFIG)
    st.plotly_chart(fig_map, use_container_width=True)

st.markdown("⚠️ **Note:** Locations are approximated based on IP data and do not represent precise physical addresses.")

# 3. Short Video Frequency Analysis
st.divider()
st.subheader("📱 Short Video Usage & Slang Proficiency")
col_freq1, col_freq2 = st.columns(2)

if not f_df.empty:
    with col_freq1:
        st.markdown("#### Distribution of Usage Frequency")
        freq_counts = f_df['Frequency'].value_counts().reset_index()
        freq_counts.columns = ['Frequency', 'Count']
        freq_counts = freq_counts.sort_values('Frequency') # Uses categorical order from cleaning
        
        fig_freq_dist = px.bar(freq_counts, x='Frequency', y='Count', color='Frequency',
                               color_discrete_sequence=px.colors.sequential.Viridis, text='Count')
        fig_freq_dist.config.update(HIGH_RES_CONFIG)
        st.plotly_chart(fig_freq_dist, use_container_width=True)

    with col_freq2:
        st.markdown("#### Average Total Score by Frequency")
        freq_score = f_df.groupby('Frequency', observed=True)['Total Score'].mean().reset_index()
        fig_freq_trend = px.line(freq_score, x='Frequency', y='Total Score', markers=True,
                                 labels={'Total Score': 'Avg Total Score'})
        fig_freq_trend.update_traces(line_color='#636EFA', fill='tozeroy') 
        fig_freq_trend.config.update(HIGH_RES_CONFIG)
        st.plotly_chart(fig_freq_trend, use_container_width=True)

st.markdown("💡 **Insight:** Users who watch short videos **Frequently** generally exhibit higher awareness and usage scores.")

# 4. Age vs Total Score
st.divider()
st.subheader("🎂 Age Group vs Slang Proficiency")

if not f_df.empty:
    # (1) 计算各年龄段的平均分（用于辅助说明）
    age_avg = f_df.groupby('Age', observed=True)['Total Score'].mean().reset_index()
    
    # (2) 创建箱线图，展示得分分布
    fig_age_score = px.box(
        f_df, 
        x='Age', 
        y='Total Score', 
        color='Age',
        points="all", # 显示所有数据点，增加散点透视感
        category_orders={"Age": ["Under 18", "18-30", "Over 30"]}, # 确保年龄轴有序
        color_discrete_sequence=px.colors.qualitative.Pastel,
        template='plotly_white',
        labels={'Total Score': 'Total Score (Awareness + Usage)'}
    )
    
    fig_age_score.update_layout(
        showlegend=False,
        xaxis_title="Age Group",
        yaxis_title="Total Score",
        height=500
    )
    fig_age_score.config.update(HIGH_RES_CONFIG)
    st.plotly_chart(fig_age_score, use_container_width=True)
    
    # 动态 Insight
    highest_age = age_avg.loc[age_avg['Total Score'].idxmax(), 'Age']
    st.markdown(f"💡 **Insight:** On average, the **{highest_age}** group shows the highest engagement with internet slang.")
    
# 5. Slang Comparative Analysis
st.divider()
st.subheader("📊 Slang Landscape: Awareness & Usage")
if not f_df.empty:
    comp_list = []
    for i in range(len(SLANG_CONTENT)):
        comp_list.append({
            "Slang": SLANG_CONTENT[i],
            "Awareness Score": f_df[f"Score_Aw_{i}"].mean(),
            "Usage Score": f_df[f"Score_Us_{i}"].mean()
        })
    fig_comp = px.scatter(pd.DataFrame(comp_list), x="Awareness Score", y="Usage Score",
                          color="Slang", text="Slang", height=500)
    fig_comp.update_traces(textposition='top center')
    fig_comp.config.update(HIGH_RES_CONFIG)
    st.plotly_chart(fig_comp, use_container_width=True)

# 6. Bar Comparison Analysis
st.divider()
st.subheader("📊 Comparative Analysis: Awareness & Usage by Slang Item")
if not f_df.empty:
    bar_data_list = []
    for i in range(len(SLANG_CONTENT)):
        bar_data_list.append({"Slang": SLANG_CONTENT[i], "Score": f_df[f"Score_Aw_{i}"].mean(), "Type": "Awareness"})
        bar_data_list.append({"Slang": SLANG_CONTENT[i], "Score": f_df[f"Score_Us_{i}"].mean(), "Type": "Usage"})

    fig_bar_comp = px.bar(pd.DataFrame(bar_data_list), x="Slang", y="Score", color="Type",
                          barmode="group", height=500, color_discrete_map={"Awareness": "#8ECAE6", "Usage": "#BDB2FF"}, template='plotly_white')
    fig_bar_comp.update_layout(xaxis_tickangle=-45)
    fig_bar_comp.config.update(HIGH_RES_CONFIG)
    st.plotly_chart(fig_bar_comp, use_container_width=True)

# 7. Individual Analysis
st.divider()
st.subheader("🔎 Individual Slang Item Analysis")
slang_idx = st.selectbox("Select Slang to Inspect:", range(len(SLANG_CONTENT)), format_func=lambda x: SLANG_CONTENT[x])

if not f_df.empty:
    # 聚合数据
    item_avg = f_df.groupby('Gender', observed=True)[[f"Score_Aw_{slang_idx}", f"Score_Us_{slang_idx}"]].mean().reset_index()
    item_avg.columns = ['Gender', 'Awareness', 'Usage']
    
    # 转换长格式以便 Plotly 绘制分组柱状图
    item_avg_melted = item_avg.melt(id_vars='Gender', var_name='Type', value_name='Score')

    # 修改后的图表：使用了低饱和度色调
    fig_ind = px.bar(
        item_avg_melted, 
        x='Gender', 
        y='Score', 
        color='Type', 
        barmode='group',
        title=f"Detailed Comparison: {SLANG_CONTENT[slang_idx]}",
        # 这里使用了更柔和的调色盘：淡蓝色和浅紫色
        color_discrete_map={"Awareness": "#B5C0D0", "Usage": "#CCD3CA"},
        template='plotly_white'
    )
    
    fig_ind.update_layout(
        yaxis_title="Average Score",
        legend_title_text='',
        height=450
    )
    fig_ind.config.update(HIGH_RES_CONFIG)
    st.plotly_chart(fig_ind, use_container_width=True)

# 8. Acquisition & Scenarios
st.divider()
st.subheader("📡 Acquisition Channels & Usage Scenarios")
col_chan, col_scene = st.columns(2)

with col_chan:
    st.markdown("#### Top Acquisition Channels")
    if not f_df.empty:
        chan_data, chan_others = process_multi_choice_with_percentages(f_df['Acquisition Channel'], CHANNEL_MAP)
        # 直接显示包含 Others 的完整统计图
        fig_chan = px.bar(chan_data.sort_values('Count', ascending=True), 
                          x='Count', y='Item', orientation='h', color='Item',
                          color_discrete_sequence=px.colors.qualitative.Pastel, template='plotly_white', height=400)
        fig_chan.config.update(HIGH_RES_CONFIG)
        st.plotly_chart(fig_chan, use_container_width=True)
        
        with st.expander("Explore 'Other' Channels"):
            if chan_others: st.write(", ".join(chan_others))
            else: st.write("No specific details provided for 'Others'.")

with col_scene:
    st.markdown("#### Usage Scenarios")
    if not f_df.empty:
        scene_data, scene_others = process_multi_choice_with_percentages(f_df['Using Scene'], SCENE_MAP)
        # 饼图中包含 Others 扇区
        fig_scene = px.pie(scene_data, values='Count', names='Item', hole=0.5,
                           color_discrete_sequence=px.colors.qualitative.Safe, template='plotly_white', height=400)
        fig_scene.config.update(HIGH_RES_CONFIG)
        st.plotly_chart(fig_scene, use_container_width=True)
        
        with st.expander("Explore 'Other' Scenarios"):
            if scene_others: st.write(", ".join(scene_others))
            else: st.write("No specific details provided for 'Others'.")

# 9. Influence
st.divider()
st.subheader("🧠 Perceived Impact on Daily Communication")
if not f_df.empty:
    impact_counts = f_df['Influence'].value_counts().reset_index()
    impact_counts.columns = ['Impact', 'Count']
    
    fig_impact = px.pie(
        impact_counts, values='Count', names='Impact', hole=0.5,
        color_discrete_sequence=px.colors.qualitative.Pastel,
        template='plotly_white'
    )
    fig_impact.update_traces(textinfo='percent+label', pull=[0.05, 0, 0])
    fig_impact.update_layout(margin=dict(t=30, b=30, l=30, r=30))
    fig_impact.config.update(HIGH_RES_CONFIG)
    st.plotly_chart(fig_impact, use_container_width=True)

# --- Footer ---
st.write("") # 增加一点留白
st.write("")
st.markdown("---") # 分割线

# 使用居中的 Markdown 替代固定定位的 HTML
st.markdown(
    """
    <div style="text-align: center; color: #A0A0A0; font-size: 14px; padding: 20px 0;">
        Made by Adele.
    </div>
    """,
    unsafe_allow_html=True
)
