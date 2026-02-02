"""
Spectral plot widget for displaying ROI spectra.
"""

import csv
from qtpy.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class SpectralPlot(QWidget):
    """Interactive spectral signature plot with matplotlib backend."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(6, 3.5), facecolor='#262626')
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ylabel = 'Radiance'
        self._setup_style()
        self._plotted_data = []
        self.wl_min, self.wl_max = 400, 2600

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)

    def set_data_type(self, data_type):
        """Set y-axis label based on data type."""
        self.ylabel = 'Reflectance' if data_type == 'reflectance' else 'Radiance'
        self.ax.set_ylabel(self.ylabel, fontsize=9)
        self.canvas.draw()

    def set_wavelength_range(self, wl_min, wl_max):
        """Set the valid wavelength range for reference lines."""
        self.wl_min, self.wl_max = wl_min, wl_max

    def _setup_style(self):
        """Configure dark theme plot styling."""
        self.ax.set_facecolor('#1a1a1a')
        self.ax.tick_params(colors='white', labelsize=8)
        self.ax.xaxis.label.set_color('white')
        self.ax.yaxis.label.set_color('white')
        self.ax.title.set_color('white')
        for spine in self.ax.spines.values():
            spine.set_color('#555555')
        self.ax.set_xlabel('Wavelength (nm)', fontsize=9)
        self.ax.set_ylabel(self.ylabel, fontsize=9)
        self.ax.set_title('ROI Spectral Signature', fontsize=10)
        self.figure.tight_layout()

    def plot_spectrum(self, wavelengths, mean_spectrum, std_spectrum=None,
                      label='ROI', color='cyan', clear=True):
        """
        Plot a spectral signature with optional standard deviation envelope.

        Args:
            wavelengths: Array of wavelength values (nm)
            mean_spectrum: Array of mean spectral values
            std_spectrum: Optional array of standard deviation values
            label: Legend label for this spectrum
            color: Line and fill color
            clear: If True, clear previous plots first
        """
        if clear:
            self.ax.clear()
            self._setup_style()
            self._plotted_data = []

        self._plotted_data.append({
            'label': label,
            'wavelengths': wavelengths,
            'mean': mean_spectrum,
            'std': std_spectrum
        })

        self.ax.plot(wavelengths, mean_spectrum, color=color, linewidth=1.2, label=label)

        if std_spectrum is not None:
            self.ax.fill_between(
                wavelengths,
                mean_spectrum - std_spectrum,
                mean_spectrum + std_spectrum,
                alpha=0.25,
                color=color
            )

        # Mark atmospheric absorption bands
        for band_start, band_end in [(1350, 1450), (1800, 1950)]:
            self.ax.axvspan(band_start, band_end, alpha=0.15, color='red')

        self.ax.legend(
            loc='upper right',
            facecolor='#333333',
            edgecolor='#555555',
            labelcolor='white',
            fontsize=8
        )
        self.ax.set_xlabel('Wavelength (nm)', fontsize=9)
        self.ax.set_ylabel(self.ylabel, fontsize=9)
        self.figure.tight_layout()
        self.canvas.draw()

    def clear_plot(self):
        """Clear all plotted data."""
        self.ax.clear()
        self._setup_style()
        self._plotted_data = []
        self.canvas.draw()

    def clear_reference_lines(self):
        """Remove all reference lines and labels."""
        for line in list(self.ax.lines):
            if hasattr(line, '_is_reference') and line._is_reference:
                line.remove()
        for txt in list(self.ax.texts):
            if hasattr(txt, '_is_reference') and txt._is_reference:
                txt.remove()

    def add_reference_line(self, wavelength, label, color='yellow', alpha=0.7):
        """
        Add a vertical reference line at a specific wavelength.

        Args:
            wavelength: Wavelength in nm
            label: Text label for the line
            color: Line and label color
            alpha: Transparency

        Returns:
            True if line was added, False if wavelength out of range
        """
        if self.wl_min <= wavelength <= self.wl_max:
            line = self.ax.axvline(
                x=wavelength, color=color, linestyle='--',
                alpha=alpha, linewidth=0.8
            )
            line._is_reference = True

            ylim = self.ax.get_ylim()
            ypos = ylim[1] * 0.95 if ylim[1] > 0 else ylim[0] * 0.05
            short_label = label[:15] + '..' if len(label) > 17 else label

            txt = self.ax.text(
                wavelength, ypos, short_label,
                fontsize=6, rotation=90, va='top', ha='right',
                color=color, alpha=0.9
            )
            txt._is_reference = True
            return True
        return False

    def draw(self):
        """Redraw the canvas."""
        self.figure.tight_layout()
        self.canvas.draw()

    def export_to_csv(self, filepath):
        """
        Export plotted spectral data to CSV.

        Args:
            filepath: Output CSV file path

        Returns:
            True if export successful, False if no data
        """
        if not self._plotted_data:
            return False

        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)

            # Build header
            header = ['Wavelength_nm']
            for data in self._plotted_data:
                header.append(f"{data['label']}_mean")
                if data['std'] is not None:
                    header.append(f"{data['label']}_std")
            writer.writerow(header)

            # Write data rows
            wavelengths = self._plotted_data[0]['wavelengths']
            for i, wl in enumerate(wavelengths):
                row = [wl]
                for data in self._plotted_data:
                    row.append(data['mean'][i])
                    if data['std'] is not None:
                        row.append(data['std'][i])
                writer.writerow(row)

        return True

    def has_spectral_data(self):
        """Check if any spectral data has been plotted."""
        return len(self._plotted_data) > 0
