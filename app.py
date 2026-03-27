from shiny.express import ui

ui.page_opts(
    title="Yearly AI Exposure Dashboard by Region",
    theme=ui.Theme.from_brand(__file__),
    fillable=True,
    lang="en",
    full_width=True,
)

with ui.navset_pill_list(id="tab", widths=(2, 10)):
    with ui.nav_panel("Occupation View", value="occupations"):
        "Main Content here"
    with ui.nav_panel("Comparison View", value="compare"):
        "Panel B content"
    with ui.nav_panel("Download", value="data_download"):
        "Panel C Content"
