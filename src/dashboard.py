import time
from typing import Tuple, List

import requests
import streamlit as st

import config
import utils


def main() -> None:
    st.set_page_config(
        page_title='ZReader',
        page_icon='📜',
        layout='wide',
        initial_sidebar_state='expanded')

    st.session_state.url = st.get_option('s3.url')  # ZreaderAPI url

    sidebar()


def sidebar() -> None:
    st.sidebar.header('Sections')

    option = st.sidebar.radio('', ('About the project', 'Inference'))

    if option == 'About the project':
        about_sidebar()
        about_page()
    else:
        is_plain, min_noise, max_noise, bw = inference_sidebar()
        inference_page(is_plain, min_noise, max_noise, bw)


def about_sidebar() -> None:
    pass


def inference_sidebar() -> Tuple[bool, int, int, int]:
    st.sidebar.text('\n')

    st.sidebar.header('Data parameters')

    data_type = st.sidebar.radio('Input type', ('Plain text', 'Noisy columns'), key='data_type',
                                 index=0 if 'Plain text' == st.session_state.get('data_type', 'Plain text') else 1)
    is_plain = data_type == 'Plain text'

    st.sidebar.text('\n')

    if is_plain:
        min_noise, max_noise = st.sidebar.slider('\nNoise range', 0, 25, key='noise_range',
                                                 value=st.session_state.get('noise_range', (0, 5)))
    else:
        min_noise, max_noise = 0, 0

    bw = st.sidebar.slider('Beam search width', 1, 26, key='beam_width',
                           value=st.session_state.get('beam_width', 3))

    return is_plain, min_noise, max_noise + 1, bw


def about_page() -> None:
    st.title('About')

    st.text('TODO: Write about the method of keyless reading in columns')


@st.cache(max_entries=26, show_spinner=False)
def get_chains(columns: List[str], bw: int) -> List[Tuple[str, float]]:
    payload = {
        'data': columns,
        'beam_width': bw
    }

    task_info = requests.post(url=f'{st.session_state.url}/zread', json=payload)
    task_info = task_info.json()

    if not task_info['task_id']:
        st.error(task_info['message'])
        st.stop()

    task_status = requests.get(url=f'{st.session_state.url}/status/{task_info["task_id"]}')
    task_status = task_status.json()

    progress_bar = st.progress(0)

    while task_status['message'] == 'Processing':
        task_status = requests.get(url=f'{st.session_state.url}/status/{task_info["task_id"]}')
        task_status = task_status.json()

        completed = task_status['progress'] or 0
        progress_bar.progress(completed / len(columns))

        time.sleep(0.3)

    requests.delete(url=f'{st.session_state.url}/status/{task_info["task_id"]}')

    return task_status['chains']


@st.cache(max_entries=1)
def get_noisy_columns(data: str, min_noise: int, max_noise: int) -> List[str]:
    columns = utils.create_noisy_columns(data, min_noise, max_noise)

    return [''.join(set(c)) for c in columns]  # shuffle columns


def inference_page(is_plain: bool, min_noise: int, max_noise: int, bw: int) -> None:
    input_label = 'Insert plain data' if is_plain else 'Insert noisy columns separated by spaces'
    st.subheader(input_label)
    data = st.text_input('', value=st.session_state.get('data', ''), key='data')

    if not data:
        st.stop()

    if is_plain:
        columns = get_noisy_columns(data, min_noise, max_noise)
    else:
        columns = utils.data_to_columns(data, separator=' ')

    st.subheader('\nColumns')
    st.text(utils.visualize_columns(columns, delimiter=''))
    st.sidebar.text('\n')

    placeholder = st.empty()

    # TODO if max_noise grater than threshold print warning about bad performance
    # TODO add history

    if columns and placeholder.button('Zread'):
        with placeholder:
            chains = get_chains(columns, bw)

        with placeholder.beta_container():
            st.subheader('\nPrediction')
            st.text('\n\n'.join(chain for chain, _ in chains))
            st.button('Clear')


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        config.project_logger.error(e)
