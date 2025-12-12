"""
MultiPage class for managing Streamlit dashboard navigation.
"""

import streamlit as st


class MultiPage:
    """
    Class to generate multiple Streamlit pages using an object-oriented approach.
    """

    def __init__(self, app_name: str) -> None:
        """
        Initialize the MultiPage app.

        Args:
            app_name: Name displayed in the browser tab
        """
        self.pages = []
        self.app_name = app_name

        # Configure page settings
        st.set_page_config(
            page_title=self.app_name,
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )

    def add_page(self, title: str, func) -> None:
        """
        Add a page to the app.

        Args:
            title: Page title shown in sidebar
            func: Function that renders the page content
        """
        self.pages.append({
            "title": title,
            "function": func
        })

    def run(self) -> None:
        """
        Run the multipage app with sidebar navigation.
        """
        # Sidebar header
        st.sidebar.title(self.app_name)
        st.sidebar.markdown("---")

        # Page selection
        page = st.sidebar.radio(
            "Navigation",
            self.pages,
            format_func=lambda page: page["title"]
        )

        # Sidebar info
        st.sidebar.markdown("---")
        st.sidebar.info(
            """
            This dashboard predicts lead conversion probability
            to help sales teams prioritise their outreach efforts.
            """
        )

        # Footer
        st.sidebar.markdown("---")
        st.sidebar.markdown(
            """
            **Lead Conversion Predictor**
            Portfolio Project 5
            Code Institute
            """
        )

        # Run selected page
        page["function"]()
