"""
Web-based hyperspectral viewer using trame.

Usage:
    # Standalone
    aviris-web
    aviris-web --port 8080 --data-dir /data

    # In JupyterLab
    from aviris_tools.web import show
    viewer = show()

    # Access data from notebook
    viewer.data_loader          # LazyHyperspectralData object
    viewer.current_spectra      # List of extracted pixel spectra
    ndvi = viewer.calculate_and_get_index('NDVI')  # numpy array
"""


def launch(port=0, data_dir="/data", host="0.0.0.0", **kwargs):
    """Launch the web viewer as a standalone app."""
    from .app import HyperspectralWebApp

    app = HyperspectralWebApp(data_dir=data_dir)
    app.server.start(port=port, host=host, open_browser=False, **kwargs)


def get_widget(data_dir="/data"):
    """
    Get a trame app instance for JupyterLab embedding.

    Returns the HyperspectralWebApp instance. Display it by returning
    `app.ui` in a notebook cell.

    Example:
        app = get_widget()
        app.ui  # displays inline in JupyterLab

        # Later, access data:
        app.data_loader  # the loaded data
        app.current_spectra  # clicked pixel spectra
    """
    from .app import HyperspectralWebApp

    app = HyperspectralWebApp(data_dir=data_dir)
    return app


def show(data_dir="/data"):
    """
    Display the viewer widget in JupyterLab.

    Returns the app instance for data access.

    Example:
        viewer = show()
        # interact with the viewer in the UI...
        # then access data:
        viewer.data_loader.wavelengths
    """
    app = get_widget(data_dir=data_dir)
    # In Jupyter, displaying app.ui renders the trame widget
    try:
        from IPython.display import display
        display(app.ui)
    except ImportError:
        pass
    return app


def main():
    """CLI entry point for aviris-web."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="aviris-web",
        description="Hyperspectral Web Viewer",
    )
    parser.add_argument("--port", type=int, default=8080, help="Server port (default: 8080)")
    parser.add_argument("--host", default="0.0.0.0", help="Server host (default: 0.0.0.0)")
    parser.add_argument("--data-dir", default="/data", help="Data directory (default: /data)")
    args = parser.parse_args()

    launch(port=args.port, host=args.host, data_dir=args.data_dir)
