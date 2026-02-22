from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Plot earthquake locations on a simple world map backdrop if available
#  - Falls back to a plain scatter plot if geopandas isn't installed.

def plot_world_scatter(df: pd.DataFrame, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import geopandas as gpd

        world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
        gdf = gpd.GeoDataFrame(
            df.copy(),
            geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
            crs="EPSG:4326",
        )

        ax = world.plot(figsize=(14, 7), linewidth=0.6)
        # size encodes magnitude a bit
        sizes = (df["magnitude"].clip(lower=0) ** 2) * 2.0
        gdf.plot(ax=ax, markersize=sizes, alpha=0.35)
        ax.set_title("Earthquake Locations (marker size ~ magnitude)")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()
        return out_path

    except Exception:
        # fallback
        plt.figure(figsize=(14, 7))
        sizes = (df["magnitude"].clip(lower=0) ** 2) * 2.0
        plt.scatter(df["longitude"], df["latitude"], s=sizes, alpha=0.35)
        plt.title("Earthquake Locations (marker size ~ magnitude)")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()
        return out_path
