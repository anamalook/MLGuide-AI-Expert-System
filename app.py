import streamlit as st
import pandas as pd
from io import StringIO

st.set_page_config(page_title="MLGuide AI | Decision Lab", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")

st.markdown('''<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html,body,[class*="css"], [data-testid="stApp"] {font-family:Inter,sans-serif;}
.stApp {background: radial-gradient(circle at 85% 0%,#142a45 0%,#081321 40%,#070f1b 100%);color:#edf6ff;}
[data-testid="stSidebar"] {background:#0d1c2d;border-right:1px solid #23415a;}
h1,h2,h3 {color:#f1f8ff !important;letter-spacing:-.025em;}
p,li,label {color:#d0e0ef;}
.hero {padding:32px 35px;border:1px solid #315b77;border-radius:20px;background:linear-gradient(110deg,#123954,#14203d 65%,#202354);margin-bottom:22px;}
.hero h1 {font-size:2.6rem;margin:8px 0;}
.eyebrow {color:#67e8f9;font-size:.85rem;letter-spacing:.16em;font-weight:800;}
.subtle {color:#a9c1d6;}
.card {background:#102237;border:1px solid #29435b;border-radius:15px;padding:20px;margin:10px 0;}
.card strong {color:#75e5f6;}
.score {font-size:1.4rem;color:#72e6fa;font-weight:800;}
[data-testid="stMetric"] {background:#102237;border:1px solid #29435b;padding:16px;border-radius:14px;}
.stButton>button[kind="primary"] {background:#06b6d4;color:#071521;border:0;font-weight:800;}
</style>''', unsafe_allow_html=True)

KB = [
    dict(name='Logistic Regression', task='Classification', sizes=['Small','Medium','Large'], patterns=['Linear','Unknown'], interpretable=True, resources=['Low','Medium','High'], why='A transparent baseline for classification; suitable when a linear decision boundary is reasonable.', caution='May underfit nonlinear patterns without feature engineering.'),
    dict(name='Decision Tree Classifier', task='Classification', sizes=['Small','Medium','Large'], patterns=['Linear','Nonlinear','Unknown'], interpretable=True, resources=['Low','Medium','High'], why='Can express nonlinear decisions as human-readable rules.', caution='Deep trees can overfit; control depth and validate.'),
    dict(name='Random Forest Classifier', task='Classification', sizes=['Small','Medium','Large'], patterns=['Linear','Nonlinear','Unknown'], interpretable=False, resources=['Medium','High'], why='Ensembles trees to capture interactions and nonlinear patterns.', caution='Less transparent than a single tree; training can require more resources.'),
    dict(name='Naive Bayes', task='Classification', sizes=['Small','Medium','Large'], patterns=['Linear','Unknown'], interpretable=True, resources=['Low','Medium','High'], why='A fast probabilistic baseline, especially useful for certain high-dimensional tasks.', caution='Conditional independence assumptions may be unrealistic.'),
    dict(name='Linear Regression', task='Regression', sizes=['Small','Medium','Large'], patterns=['Linear','Unknown'], interpretable=True, resources=['Low','Medium','High'], why='A simple interpretable baseline for continuous numeric targets.', caution='Needs diagnostic checks for assumptions and nonlinear relationships.'),
    dict(name='Decision Tree Regressor', task='Regression', sizes=['Small','Medium','Large'], patterns=['Linear','Nonlinear','Unknown'], interpretable=True, resources=['Low','Medium','High'], why='Models nonlinear numeric outcomes with understandable splits.', caution='Can overfit and extrapolates poorly beyond observed targets.'),
    dict(name='Random Forest Regressor', task='Regression', sizes=['Small','Medium','Large'], patterns=['Linear','Nonlinear','Unknown'], interpretable=False, resources=['Medium','High'], why='Combines many trees for nonlinear regression problems.', caution='Uses more compute and cannot reliably extrapolate outside training range.'),
    dict(name='K-Means', task='Clustering', sizes=['Small','Medium','Large'], patterns=['Linear','Unknown'], interpretable=True, resources=['Low','Medium','High'], why='A practical baseline for compact, roughly spherical clusters.', caution='Requires choosing the number of clusters and scaling features.'),
    dict(name='DBSCAN', task='Clustering', sizes=['Small','Medium'], patterns=['Nonlinear','Unknown'], interpretable=False, resources=['Medium','High'], why='Can find irregular density-based groups and flag noise.', caution='Sensitive to distance scale and density parameters; may be expensive on large datasets.'),
    dict(name='Agglomerative Clustering', task='Clustering', sizes=['Small','Medium'], patterns=['Nonlinear','Unknown'], interpretable=True, resources=['Medium','High'], why='Produces a hierarchical view of group structure.', caution='Can be expensive in time and memory on large datasets.'),
]

def recommend(task, size, pattern, explain, resources):
    results=[]
    for item in KB:
        if item['task'] != task:
            continue
        score=0
        matched=[]
        if size in item['sizes']:
            score+=2;matched.append('Dataset-size heuristic matched (+2)')
        if pattern in item['patterns']:
            score+=2;matched.append('Expected pattern matched (+2)')
        if explain == 'High priority' and item['interpretable']:
            score+=2;matched.append('Interpretability preference matched (+2)')
        if explain == 'Flexible':
            score+=1;matched.append('No strict interpretability requirement (+1)')
        if resources in item['resources']:
            score+=2;matched.append('Compute budget heuristic matched (+2)')
        results.append({**item,'score':score,'matched':matched})
    return sorted(results,key=lambda x:(-x['score'],x['name']))[:3]

with st.sidebar:
    st.markdown('## 🧠 MLGuide AI')
    st.caption('INTELLIGENT ML DECISION LAB')
    st.divider()
    page=st.radio('WORKSPACE', ['🏠 Overview','📂 Dataset Explorer','🧠 Algorithm Advisor','ℹ️ About the Expert System'])
    st.divider()
    st.caption('Python · Streamlit · Pandas')
    st.caption('Educational decision-support prototype')

if page=='🏠 Overview':
    st.markdown('<div class="hero"><div class="eyebrow">WELCOME TO THE DECISION LAB</div><h1>Choose smarter ML starting points.</h1><p class="subtle">Explore a dataset, answer a few questions, and discover rule-based algorithm suggestions with transparent explanations.</p></div>',unsafe_allow_html=True)
    a,b,c=st.columns(3)
    with a: st.metric('Supported ML tasks','3')
    with b: st.metric('Algorithms in knowledge base',str(len(KB)))
    with c: st.metric('Recommendation approach','IF–THEN')
    st.markdown('### Your workflow')
    st.markdown('**01 · Explore data** — Upload a CSV and review its basic structure.\n\n**02 · Describe the task** — Choose task type, approximate size, pattern, interpretability, and compute budget.\n\n**03 · Understand suggestions** — Review ranked rule matches, limitations, and evaluation guidance.')
    st.info('This is an educational expert system. It does not train models, benchmark performance, or guarantee the best algorithm.')

elif page=='📂 Dataset Explorer':
    st.title('📂 Dataset Explorer')
    st.write('Upload a CSV file to inspect its structure before selecting an ML approach.')
    uploaded=st.file_uploader('Choose a CSV dataset',type=['csv'])
    if uploaded is None:
        st.info('No dataset uploaded yet. You can still use the Algorithm Advisor without a file.')
    else:
        try:
            df=pd.read_csv(uploaded)
            if df.empty or len(df.columns)==0:
                st.warning('This CSV has no usable rows or columns.')
            else:
                a,b,c,d=st.columns(4)
                a.metric('Rows',f'{len(df):,}')
                b.metric('Columns',len(df.columns))
                c.metric('Missing cells',f'{int(df.isna().sum().sum()):,}')
                d.metric('Numeric columns',len(df.select_dtypes(include='number').columns))
                st.subheader('Data preview')
                st.dataframe(df.head(50),use_container_width=True)
                st.subheader('Column health')
                summary=pd.DataFrame({'Column':df.columns,'Data type':[str(t) for t in df.dtypes], 'Missing values':df.isna().sum().values,'Unique values':df.nunique(dropna=True).values})
                st.dataframe(summary,use_container_width=True,hide_index=True)
                numeric=list(df.select_dtypes(include='number').columns)
                if numeric:
                    st.subheader('Numeric feature distribution')
                    feature=st.selectbox('Select numeric column',numeric)
                    st.bar_chart(df[feature].dropna().value_counts(bins=12).sort_index())
                    st.caption('Binned value counts provide a simple distribution preview.')
                st.caption('Dataset preview is descriptive only. No algorithm is trained or evaluated here.')
        except Exception as e:
            st.error('Could not read this CSV. Check that it is a valid, consistently formatted CSV file.')
            st.caption(str(e))

elif page=='🧠 Algorithm Advisor':
    st.title('🧠 Algorithm Advisor')
    st.write('Set your requirements and run the transparent rule-based inference engine.')
    with st.form('advisor'):
        left,right=st.columns(2)
        with left:
            task=st.selectbox('ML task',['Classification','Regression','Clustering'])
            size=st.selectbox('Dataset size',['Small','Medium','Large'],index=1)
            pattern=st.selectbox('Expected pattern',['Unknown','Linear','Nonlinear'])
        with right:
            explain=st.selectbox('Interpretability',['High priority','Flexible'])
            resources=st.selectbox('Compute budget',['Low','Medium','High'],index=1)
        submitted=st.form_submit_button('✨ Generate recommendations',type='primary',use_container_width=True)
    if submitted:
        st.session_state['advisor_result']={'task':task,'size':size,'pattern':pattern,'explain':explain,'resources':resources,'results':recommend(task,size,pattern,explain,resources)}
    if 'advisor_result' in st.session_state:
        state=st.session_state['advisor_result']
        st.divider()
        st.subheader('Your top 3 starting points')
        for rank,item in enumerate(state['results'],1):
            with st.container(border=True):
                a,b=st.columns([4,1])
                with a:
                    st.markdown(f'### {rank:02d} · {item["name"]}')
                    st.write(item['why'])
                with b: st.metric('Rule-match points',f'{item["score"]}/8' if state['explain']=='High priority' else f'{item["score"]}/7')
                st.markdown('**Why the rules selected it**')
                if item['matched']:
                    for match in item['matched']: st.write('✓ '+match)
                else: st.write('No additional preference rules matched.')
                st.warning('**Consideration:** '+item['caution'])
        st.subheader('Comparison')
        table=pd.DataFrame([{'Algorithm':i['name'],'Rule-match points':i['score'],'Interpretable heuristic':'Yes' if i['interpretable'] else 'Limited','Caution':i['caution']} for i in state['results']])
        st.dataframe(table,hide_index=True,use_container_width=True)
        report=StringIO()
        report.write('MLGuide AI - Rule-Based Algorithm Advisor\n')
        for field in ['task','size','pattern','explain','resources']:
            report.write(f'{field.title()}: {state[field]}\n')
        report.write('\nRule-match points are not accuracy or probabilities.\n')
        for i in state['results']:
            report.write(f'\n{i["name"]} | Points: {i["score"]}\nWhy: {i["why"]}\nCaution: {i["caution"]}\n')
        st.download_button('⬇️ Download recommendation report',report.getvalue(),file_name='MLGuide_recommendations.txt',mime='text/plain')
        st.info('Next step: use cross-validation and task-appropriate metrics on real data to compare candidate models. This tool does not evaluate actual performance.')

else:
    st.title('ℹ️ About MLGuide AI')
    st.markdown('''### Project objective
Recommend **starting-point algorithms** for common machine-learning tasks using explicit, explainable heuristics.

### Expert system architecture
**Knowledge base:** Python records describing supported tasks, data patterns, resource heuristics, interpretability, and cautions.

**Inference engine:** A rule matcher that filters by task, awards points for matching preferences, and ranks up to three options.

**Explanation facility:** Displays matched rules, reasons, and limitations for each recommendation.

**User interface:** A Streamlit dashboard with dataset inspection and downloadable reports.

### Important limitations
This is **not** a trained machine-learning model or an automated benchmark. Dataset size and computing-cost labels are approximate educational heuristics, not universal facts. Uploaded CSV files are inspected only; the recommendation rules do not automatically infer target type, task, feature engineering needs, or model accuracy from the file.

### Suggested future improvements
Add data-aware feature checks, a broader algorithm knowledge base, validation tests, and actual cross-validated model comparisons.''')
