import time
from http import HTTPStatus
from typing import Tuple, List

import requests
import streamlit as st
from requests.exceptions import ConnectionError

from config import var
from zreader.utils.data import data_to_columns, create_noisy_columns
from zreader.utils.visualization import visualize_columns


def main() -> None:
    st.set_page_config(
        page_title='ZReader',
        page_icon='📜',
        layout='wide',
        initial_sidebar_state='expanded')

    if 'history' not in st.session_state:
        st.session_state.history = list()

    if 'regenerate' not in st.session_state:
        st.session_state.regenerate = False

    if 'columns' not in st.session_state:
        st.session_state.columns = None

    sidebar()


def set_regenerate() -> None:
    st.session_state.regenerate = True


def unset_regenerate() -> None:
    st.session_state.regenerate = False


def sidebar() -> None:
    st.sidebar.header('Sections')

    option = st.sidebar.radio('', ('Inference', 'Inference history'))

    if option == 'Inference':
        is_plain, min_noise, max_noise, bw = inference_sidebar()
        inference_page(is_plain, min_noise, max_noise, bw)
    else:
        history_sidebar()
        history_page()


def inference_sidebar() -> Tuple[bool, int, int, int]:
    st.sidebar.text('\n')

    st.sidebar.header('Data parameters')

    data_type = st.sidebar.radio('Input type', ('Plain text', 'Noisy columns'), key='data_type',
                                 index=0 if 'Plain text' == st.session_state.get('data_type', 'Plain text') else 1)
    is_plain = data_type == 'Plain text'

    st.sidebar.text('\n')

    if is_plain:
        min_noise, max_noise = st.sidebar.slider('\nNoise range', 0, 25, key='noise_range',
                                                 value=st.session_state.get('noise_range', (0, 5)),
                                                 on_change=set_regenerate)
    else:
        min_noise, max_noise = 0, 0

    bw = st.sidebar.slider('Beam search width', 1, 26, key='beam_width',
                           value=st.session_state.get('beam_width', 5))

    if max_noise > var.MAX_NOISE:
        st.sidebar.warning('Max noise value is too large. This will entail poor performance')

    return is_plain, min_noise, max_noise + 1, bw


def history_sidebar() -> None:
    pass


def save_to_history(is_plain: bool,
                    min_noise: int,
                    max_noise: int,
                    bw: int,
                    columns: List[str],
                    chains: List[Tuple[str, float]]
                    ) -> None:
    text = st.session_state.data if is_plain else None

    st.session_state.history.append((is_plain, text, min_noise, max_noise, bw, columns, chains))


@st.cache(ttl=3600, show_spinner=False, suppress_st_warning=True)
def predict(columns: List[str], bw: int) -> List[Tuple[str, float]]:
    try:
        payload = {
            'data': columns,
            'beam_width': bw
        }

        task_info = requests.post(url=f'{var.FASTAPI_URL}/zread', json=payload)
        task_info = task_info.json()

        if not task_info.get('task_id'):
            st.error(task_info)
            st.stop()

        task_status = requests.get(url=f'{var.FASTAPI_URL}/status/{task_info["task_id"]}')
        task_status = task_status.json()

        progress_bar = st.progress(0)

        while task_status['status_code'] == HTTPStatus.PROCESSING:
            task_status = requests.get(url=f'{var.FASTAPI_URL}/status/{task_info["task_id"]}')
            task_status = task_status.json()

            completed = task_status['progress'] or 0
            progress_bar.progress(completed / len(columns))

            time.sleep(0.3)

        requests.delete(url=f'{var.FASTAPI_URL}/status/{task_info["task_id"]}')

        if task_status['status_code'] != HTTPStatus.OK:
            st.error(task_status['message'])
            st.stop()

        return task_status['chains']

    except ConnectionError:
        st.error(f'It seems that the API service is not running.\n\n'
                 f'Failed to establish a {var.FASTAPI_URL} connection.')
        st.stop()


def get_noisy_columns(data: str, min_noise: int, max_noise: int) -> List[str]:
    columns = create_noisy_columns(data, min_noise, max_noise)

    return [''.join(set(c)) for c in columns]  # kinda shuffle columns


def inference_page(is_plain: bool, min_noise: int, max_noise: int, bw: int) -> None:
    input_label = 'Insert plain data' if is_plain else 'Insert noisy columns separated by spaces'
    st.subheader(input_label)
    data = st.text_input('', value=st.session_state.get('data', ''), key='data')

    if not data:
        st.stop()

    if st.session_state.regenerate or not st.session_state.columns:
        if is_plain:
            columns = get_noisy_columns(data, min_noise, max_noise)
        else:
            columns = data_to_columns(data, separator=' ')

        st.session_state.columns = columns
        unset_regenerate()

    else:
        columns = st.session_state.columns

    st.subheader('\nColumns')
    st.text(visualize_columns(columns, delimiter=''))
    st.subheader('\n')

    placeholder = st.empty()
    zread_field, regen_filed = placeholder.columns([.07, 1])
    regen_filed.button('Regenerate', on_click=set_regenerate)

    if columns and zread_field.button('Zread'):
        with placeholder:
            chains = predict(columns, bw)

        with placeholder.container():
            st.subheader('\nPrediction')
            st.text('\n\n'.join(chain for chain, _ in chains))
            st.button('Clear')

        save_to_history(is_plain, min_noise, max_noise, bw, columns, chains)


def history_page() -> None:
    st.header('Inference History')

    if len(st.session_state.history) == 0:
        st.info('No records saved')
        return

    for record_id, (is_plain, text, min_noise, max_noise, bw, columns, chains) in enumerate(st.session_state.history,
                                                                                            start=1):
        st.info(f'Record {record_id}')

        if is_plain:
            st.text(f'Plain data: {text}')

        st.text(f'Noise range: [{min_noise}, {max_noise}]')
        st.text(f'Beam search width: {bw}')

        st.text('Columns:')
        st.text(visualize_columns(columns, delimiter=''))

        st.text('Prediction:')
        st.text('\n\n'.join(chain for chain, _ in chains))


if __name__ == '__main__':
    main()