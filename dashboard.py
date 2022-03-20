import streamlit as st


def main() -> None:
    st.set_page_config(
        page_title='ZReader',
        page_icon='📜',
        layout='wide',
        initial_sidebar_state='expanded')

    st.info('Test GitHub actions')


if __name__ == '__main__':
    main()
