import glob
import functools
import streamlit as st
import os
import omegaconf
import pandas as pd
import plotly.graph_objects as go


chart = functools.partial(st.plotly_chart, use_container_width=True)
OUTPUTS_PATH = "../../outputs"


@st.experimental_memo
def generate_general_results(base_path: str, versions: list, attributes: list):
    names = [name.split('/')[-1] for name in glob.glob(os.path.join(base_path, versions[0], '*M-Acc.csv'))]
    output = dict()
    for name in names:
        df = pd.DataFrame(columns=['version'] + attributes, dtype=object)
        for version in versions:
            row = pd.read_csv(glob.glob(os.path.join(base_path, version, name))[0], index_col=0).loc[0]
            df.loc[len(df.index)] = [version] + list(row)
        output[name] = df
    return output


def main() -> None:
    exp_name = st.selectbox(
        'Выберите эксперимент',
        os.listdir(OUTPUTS_PATH))

    # Config info
    config = omegaconf.OmegaConf.load(f'{OUTPUTS_PATH}/{exp_name}/multirun.yaml')

    st.markdown(f"##### Данные")
    st.json(omegaconf.OmegaConf.to_container(config["Data"]), expanded=False)

    st.markdown(f"##### Базовые параметры")
    st.json(omegaconf.OmegaConf.to_container(config["version"]), expanded=False)

    base_path = f"{OUTPUTS_PATH}/{exp_name}"
    versions = next(os.walk(base_path))[1]

    for i, version in enumerate(versions):
        st.markdown(f"##### Эксперимент #{i}")
        dir_path = os.path.join(base_path, version)
        st.write(omegaconf.OmegaConf.load(f'{dir_path}/.hydra/overrides.yaml'))

    # Results
    name = glob.glob(os.path.join(base_path, versions[0], '*M-Acc.csv'))[0]
    attributes = pd.read_csv(name, index_col=0).columns.tolist()
    stats = generate_general_results(base_path, versions, attributes)

    for name, stat in stats.items():
        st.markdown(f"##### {name}")
        st.write(stat)

    for name, stat in stats.items():
        st.markdown(f"##### {name}")
        fig = go.Figure(data=[
            go.Bar(name=row["version"], x=attributes, y=[row[att] for att in attributes])
            for index, row in stat.iterrows()
        ])
        fig.update_layout(barmode='group')
        chart(fig)


if __name__ == "__main__":
    st.set_page_config(
        "Attributes: результаты",
        "📊",
        initial_sidebar_state="expanded",
        layout="wide",
    )
    main()
