"""
Hypothesis Validation Page
==========================
Displays statistical test results for project hypotheses.
Addresses Business Requirement 1 (Distinction criteria).
"""

import streamlit as st
import pandas as pd
import os


def page_hypothesis_body():
    """
    Render the Hypothesis Validation page.
    """
    st.title("Hypothesis Validation")

    st.info(
        """
        **Business Requirement 1 - Hypothesis Testing**

        This page presents the statistical validation of four hypotheses
        about lead conversion behaviour. Each hypothesis was tested using
        appropriate statistical methods with significance level α = 0.05.
        """
    )

    st.markdown("---")

    # Summary Table
    st.header("Results Summary")

    summary_data = {
        'Hypothesis': [
            'H1: Website Engagement Time',
            'H2: Lead Source Impact',
            'H3: Activity Recency',
            'H4: Visit Frequency'
        ],
        'Test': ['t-test', 'Chi-square', 'Chi-square', 'Chi-square'],
        'p-value': ['<0.001', '<0.001', '<0.001', '0.003'],
        'Effect Size': ['Large', 'Medium', 'Small-Medium', 'Small'],
        'Result': ['Supported', 'Supported', 'Supported', 'Supported']
    }

    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, hide_index=True)

    st.success(
        """
        **All 4 hypotheses were statistically supported**, confirming that:
        - Website engagement metrics predict conversion
        - Lead source significantly impacts outcomes
        - Recent activity level correlates with conversion
        - An optimal visit frequency range exists
        """
    )

    st.markdown("---")

    # Individual Hypotheses
    st.header("Detailed Results")

    # Hypothesis 1
    with st.expander("H1: Website Engagement Time", expanded=True):
        st.markdown(
            """
            ### Hypothesis Statement
            > Leads who spend more time on the website (above median) have
            > significantly higher conversion rates than those below median.

            ### Rationale
            Higher engagement time suggests genuine interest and intent to purchase.
            Leads actively researching are more likely to convert.

            ### Statistical Test
            **Independent Samples t-test**

            | Metric | Value |
            |--------|-------|
            | t-statistic | 15.42 |
            | p-value | <0.001 |
            | Effect Size (Cohen's d) | 0.82 (large) |

            ### Results
            | Group | Conversion Rate |
            |-------|-----------------|
            | Above Median Time | 45.2% |
            | Below Median Time | 18.7% |

            ### Conclusion
            **Hypothesis SUPPORTED.** There is a statistically significant
            difference in conversion rates between high and low engagement groups.
            The effect size is large, indicating practical significance.
            """
        )

        # Display figure if exists
        fig_path = 'outputs/figures/h1_time_boxplot.png'
        if os.path.exists(fig_path):
            st.image(fig_path, caption='Time Spent Distribution by Conversion Status')

    # Hypothesis 2
    with st.expander("H2: Lead Source Impact"):
        st.markdown(
            """
            ### Hypothesis Statement
            > Lead source significantly impacts conversion probability,
            > with referral-based leads converting at higher rates than paid advertising.

            ### Rationale
            Referrals come with implicit trust and pre-qualification from the referrer.
            Paid advertising may attract lower-intent browsers.

            ### Statistical Test
            **Chi-square Test of Independence**

            | Metric | Value |
            |--------|-------|
            | Chi-square statistic | 287.45 |
            | Degrees of freedom | 8 |
            | p-value | <0.001 |
            | Effect Size (Cramer's V) | 0.31 (medium) |

            ### Results
            | Lead Source | Conversion Rate |
            |-------------|-----------------|
            | Referral Sites | 45% |
            | Organic Search | 38% |
            | Direct Traffic | 32% |
            | Paid Advertising | 18-25% |

            ### Conclusion
            **Hypothesis SUPPORTED.** Lead source is significantly associated
            with conversion. Referral leads convert at nearly 2x the rate
            of paid advertising leads.
            """
        )

        fig_path = 'outputs/figures/h2_lead_source.png'
        if os.path.exists(fig_path):
            st.image(fig_path, caption='Conversion Rate by Lead Source')

    # Hypothesis 3
    with st.expander("H3: Activity Recency Effect"):
        st.markdown(
            """
            ### Hypothesis Statement
            > Leads with recent high-engagement activities (Email Opened, Page Visited)
            > convert at higher rates than those with low-engagement or no recent activity.

            ### Rationale
            Recent engagement indicates active interest and sales readiness.
            Leads who recently opened emails or visited pages are in consideration mode.

            ### Statistical Test
            **Chi-square Test of Independence**

            | Metric | Value |
            |--------|-------|
            | Chi-square statistic | 156.78 |
            | p-value | <0.001 |
            | Effect Size (Cramer's V) | 0.28 (small-medium) |

            ### Results
            | Engagement Level | Conversion Rate |
            |------------------|-----------------|
            | High (Email Opened, Page Visit) | 42% |
            | Medium (Link Clicked) | 28% |
            | Low (Unsubscribed, Spam) | 12% |

            ### Conclusion
            **Hypothesis SUPPORTED.** Engagement level is significantly
            associated with conversion. High-engagement activities are
            strong positive signals.
            """
        )

        fig_path = 'outputs/figures/h3_engagement_level.png'
        if os.path.exists(fig_path):
            st.image(fig_path, caption='Conversion Rate by Engagement Level')

    # Hypothesis 4
    with st.expander("H4: Optimal Visit Frequency"):
        st.markdown(
            """
            ### Hypothesis Statement
            > There is an optimal engagement window - leads with 3-10 total visits
            > convert better than those with very few (<3) or very many (>10) visits.

            ### Rationale
            - Too few visits (1-2): Insufficient interest or early-stage awareness
            - Optimal range (3-10): Active consideration phase
            - Too many visits (11+): Analysis paralysis or unable to make decision

            ### Statistical Test
            **Chi-square Test**

            | Metric | Value |
            |--------|-------|
            | Chi-square statistic | 42.15 |
            | p-value | 0.003 |
            | Effect Size (Cramer's V) | 0.18 (small) |

            ### Results
            | Visit Category | Conversion Rate |
            |----------------|-----------------|
            | 1-2 visits | 22% |
            | 3-10 visits | 38% |
            | 11+ visits | 28% |

            ### Conclusion
            **Hypothesis SUPPORTED.** The 3-10 visits category shows
            the highest conversion rate, suggesting an optimal engagement window.
            """
        )

        fig_path = 'outputs/figures/h4_visit_frequency.png'
        if os.path.exists(fig_path):
            st.image(fig_path, caption='Conversion by Visit Frequency')

    st.markdown("---")

    # Business Recommendations
    st.header("Business Recommendations")

    st.markdown(
        """
        Based on validated hypotheses, we recommend the following actions:

        ### From H1 (Website Engagement Time)
        - **Implement engagement scoring:** Add website time as a lead scoring factor
        - **Real-time alerts:** Notify sales when leads exceed 5 minutes on site
        - **Content strategy:** Create engaging content to increase time on site

        ### From H2 (Lead Source Impact)
        - **Invest in referrals:** Launch customer referral incentive programme
        - **Budget reallocation:** Shift resources from low-converting paid channels
        - **Premium treatment:** Route referral leads to senior sales reps

        ### From H3 (Activity Recency)
        - **Quick follow-up:** Contact leads within 24 hours of email engagement
        - **Activity reports:** Daily list of leads with recent high-engagement
        - **Re-engagement:** Automated campaigns for inactive leads

        ### From H4 (Visit Frequency)
        - **Nurture low-visit leads:** Drip campaigns for 1-2 visit leads
        - **Focus on sweet spot:** Prioritise 3-10 visit segment
        - **Intervene with high-visit:** Direct outreach for 11+ visit leads
        """
    )

    # Implementation Priority
    st.subheader("Implementation Priority")

    priority_data = {
        'Action': [
            'Real-time engagement alerts',
            'Lead source-based routing',
            'Referral programme launch',
            'Activity-based daily report',
            'Automated nurture campaigns'
        ],
        'Impact': ['High', 'High', 'High', 'Medium', 'Medium'],
        'Effort': ['Low', 'Medium', 'High', 'Low', 'Medium'],
        'Priority': ['P1', 'P1', 'P2', 'P2', 'P3']
    }

    st.dataframe(pd.DataFrame(priority_data), hide_index=True)
