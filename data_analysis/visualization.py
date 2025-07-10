import pandas as pd
import holoviews as hv
import datashader as ds
from holoviews.operation.datashader import datashade, dynspread
import panel as pn
import colorcet as cc

hv.extension("bokeh")
pn.extension()

# Load the full CSV
df = pd.read_csv("NPs_BHVO_Oct23_full.csv")

# One HoloViews Curve per ion, all over-laid
ion_cols = df.columns[1:]
overlay   = hv.NdOverlay({
    ion: hv.Curve((df["Time (ms)"], df[ion]), label=ion)
    for ion in ion_cols
}, kdims="Ion")

# Distinct colour for each isotope
palette   = cc.glasbey_light            # 256 visually distinct colours
color_key = {ion: palette[i % len(palette)] for i, ion in enumerate(ion_cols)}

# Datashader rasterises on every zoom; no explicit aggregator needed
shaded = datashade(
    overlay,
    color_key=color_key,
    width=1400,
    height=800
)

# Make thin lines easier to see
final = dynspread(shaded, threshold=0.5, max_px=4)

# Full-window Panel pane (interactive Bokeh toolbar included)
plot_panel = pn.panel(final, sizing_mode="stretch_both")

# Save HTML (+ _files dir) **without** embed=True, so it stays interactive
plot_panel.save("ion_interactive_fullscreen.html", resources="inline")
