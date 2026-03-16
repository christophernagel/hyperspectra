"""
VTK-based 3D pipeline for hyperspectral data cube visualization.

Uses VtkLocalView (vtk.js) for client-side WebGL rendering — no server GPU needed.
Provides volume rendering, interactive slice planes, and outline wireframe.
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)

# VTK imports — grouped to make import errors clear
from vtkmodules.vtkCommonCore import vtkFloatArray, vtkUnsignedCharArray
from vtkmodules.util.numpy_support import numpy_to_vtk
from vtkmodules.vtkCommonDataModel import (
    vtkImageData,
    vtkPlane,
)
from vtkmodules.vtkFiltersCore import vtkCutter
from vtkmodules.vtkFiltersModeling import vtkOutlineFilter
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkDataSetMapper,
    vtkPolyDataMapper,
    vtkRenderer,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkColorTransferFunction,
    vtkVolume,
    vtkVolumeProperty,
)
from vtkmodules.vtkRenderingVolume import vtkFixedPointVolumeRayCastMapper
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera

# Required for rendering factory registration
import vtkmodules.vtkRenderingOpenGL2  # noqa: F401
import vtkmodules.vtkRenderingVolumeOpenGL2  # noqa: F401

from vtkmodules.vtkCommonDataModel import vtkPiecewiseFunction


# Colormap presets: name -> list of (value, r, g, b) normalized to 0-1
_COLORMAPS = {
    "Viridis": [
        (0.0, 0.267, 0.004, 0.329),
        (0.25, 0.282, 0.140, 0.458),
        (0.5, 0.127, 0.566, 0.551),
        (0.75, 0.544, 0.773, 0.248),
        (1.0, 0.993, 0.906, 0.144),
    ],
    "Plasma": [
        (0.0, 0.050, 0.030, 0.528),
        (0.25, 0.494, 0.012, 0.658),
        (0.5, 0.798, 0.280, 0.470),
        (0.75, 0.973, 0.585, 0.254),
        (1.0, 0.940, 0.975, 0.131),
    ],
    "Inferno": [
        (0.0, 0.001, 0.000, 0.014),
        (0.25, 0.341, 0.062, 0.429),
        (0.5, 0.735, 0.216, 0.330),
        (0.75, 0.978, 0.557, 0.035),
        (1.0, 0.988, 0.998, 0.645),
    ],
    "Grayscale": [
        (0.0, 0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0, 1.0),
    ],
}


class VtkCubeViewer:
    """
    Manages the VTK rendering pipeline for 3D hyperspectral cube visualization.

    The pipeline supports:
    - Volume rendering (GPU ray casting via vtk.js WebGL)
    - 3 orthogonal slice planes (vtkCutter)
    - Wireframe outline of the cube bounds
    - Configurable transfer functions (color + opacity)
    """

    def __init__(self):
        self.renderer = vtkRenderer()
        self.render_window = vtkRenderWindow()
        self.render_window.SetOffScreenRendering(1)
        self.render_window.AddRenderer(self.renderer)

        self.interactor = vtkRenderWindowInteractor()
        self.interactor.SetRenderWindow(self.render_window)
        style = vtkInteractorStyleTrackballCamera()
        self.interactor.SetInteractorStyle(style)

        # Data
        self._image_data = None
        self._cube_shape = None  # (n_bands, n_rows, n_cols)
        self._vmin = 0.0
        self._vmax = 1.0

        # Actors
        self._volume_actor = None
        self._outline_actor = None
        self._slice_actors = {}  # "x", "y", "z" -> vtkActor
        self._slice_cutters = {}  # "x", "y", "z" -> vtkCutter
        self._slice_planes = {}  # "x", "y", "z" -> vtkPlane

        # State
        self._volume_visible = False

        self.renderer.SetBackground(0.12, 0.12, 0.12)

    def load_cube(self, cube, wavelengths, vmin=None, vmax=None):
        """
        Load a numpy cube into the VTK pipeline.

        Parameters
        ----------
        cube : np.ndarray
            Shape (n_bands, n_rows, n_cols), float32.
        wavelengths : np.ndarray
            1D wavelength array.
        vmin, vmax : float or None
            Value range for color mapping. Auto-computed if None.
        """
        n_bands, n_rows, n_cols = cube.shape
        self._cube_shape = (n_bands, n_rows, n_cols)

        # Auto-compute range
        if vmin is None or vmax is None:
            flat = cube[np.isfinite(cube)]
            if len(flat) > 100000:
                flat = np.random.choice(flat, 100000, replace=False)
            vmin = float(np.percentile(flat, 2))
            vmax = float(np.percentile(flat, 98))
        self._vmin = vmin
        self._vmax = vmax

        # Clamp and normalize to uint8 for efficient transfer
        clamped = np.clip(np.nan_to_num(cube, nan=vmin), vmin, vmax)
        normalized = ((clamped - vmin) / max(vmax - vmin, 1e-10) * 255).astype(np.uint8)

        # VTK SetDimensions(nx, ny, nz) uses x-fastest point ordering:
        #   point_id = x + y*nx + z*nx*ny
        # Our cube is (n_bands, n_rows, n_cols) — C-order ravel gives:
        #   index = band*n_rows*n_cols + row*n_cols + col
        # which matches VTK's ordering for dims (n_cols, n_rows, n_bands).
        # No transpose needed.
        vtk_flat = np.ascontiguousarray(normalized).ravel()

        # Create vtkImageData
        image_data = vtkImageData()
        image_data.SetDimensions(n_cols, n_rows, n_bands)
        image_data.SetSpacing(1.0, 1.0, 1.0)
        image_data.SetOrigin(0.0, 0.0, 0.0)

        # Deep copy into VTK-owned memory to avoid numpy GC issues
        scalars = numpy_to_vtk(vtk_flat, deep=True)
        scalars.SetName("Reflectance")
        image_data.GetPointData().SetScalars(scalars)

        self._image_data = image_data

        # Clear old actors
        self.renderer.RemoveAllViewProps()
        self._slice_actors.clear()
        self._slice_cutters.clear()
        self._slice_planes.clear()
        self._volume_actor = None
        self._outline_actor = None

        # Add outline
        self._add_outline()

        # Add default slice planes at midpoints
        self._add_slice("x", n_cols // 2)
        self._add_slice("y", n_rows // 2)
        self._add_slice("z", n_bands // 2)

        self.renderer.ResetCamera()

        logger.info(
            f"VTK cube loaded: {n_bands}x{n_rows}x{n_cols}, "
            f"range [{vmin:.4f}, {vmax:.4f}]"
        )

    def _add_outline(self):
        """Add wireframe bounding box."""
        outline_filter = vtkOutlineFilter()
        outline_filter.SetInputData(self._image_data)

        mapper = vtkPolyDataMapper()
        mapper.SetInputConnection(outline_filter.GetOutputPort())

        actor = vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0.7, 0.7, 0.7)
        actor.GetProperty().SetLineWidth(1.5)

        self.renderer.AddActor(actor)
        self._outline_actor = actor

    def _add_slice(self, axis, index):
        """
        Add or update a slice plane along the given axis.

        Parameters
        ----------
        axis : str
            'x' (col), 'y' (row), or 'z' (band/wavelength).
        index : int
            Position along the axis.
        """
        if self._image_data is None:
            return

        # Define the cutting plane
        plane = vtkPlane()
        origin = [0.0, 0.0, 0.0]
        normal = [0.0, 0.0, 0.0]

        if axis == "x":
            origin[0] = float(index)
            normal[0] = 1.0
        elif axis == "y":
            origin[1] = float(index)
            normal[1] = 1.0
        elif axis == "z":
            origin[2] = float(index)
            normal[2] = 1.0

        plane.SetOrigin(*origin)
        plane.SetNormal(*normal)

        # Create cutter
        cutter = vtkCutter()
        cutter.SetCutFunction(plane)
        cutter.SetInputData(self._image_data)

        mapper = vtkDataSetMapper()
        mapper.SetInputConnection(cutter.GetOutputPort())
        mapper.SetScalarRange(0, 255)
        mapper.SetLookupTable(self._build_lut())

        actor = vtkActor()
        actor.SetMapper(mapper)

        # Remove old actor for this axis if exists
        if axis in self._slice_actors:
            self.renderer.RemoveActor(self._slice_actors[axis])

        self.renderer.AddActor(actor)
        self._slice_actors[axis] = actor
        self._slice_cutters[axis] = cutter
        self._slice_planes[axis] = plane

    def set_slice_position(self, axis, index):
        """Move an existing slice plane to a new position."""
        if axis not in self._slice_planes:
            self._add_slice(axis, index)
            return

        plane = self._slice_planes[axis]
        origin = list(plane.GetOrigin())
        if axis == "x":
            origin[0] = float(index)
        elif axis == "y":
            origin[1] = float(index)
        elif axis == "z":
            origin[2] = float(index)
        plane.SetOrigin(*origin)
        self._slice_cutters[axis].Update()

    def set_slice_visible(self, axis, visible):
        """Show or hide a slice plane."""
        if axis in self._slice_actors:
            self._slice_actors[axis].SetVisibility(visible)

    def set_colormap(self, colormap_name="Viridis"):
        """Update the color lookup table for slice planes."""
        lut = self._build_lut(colormap_name)
        for axis, actor in self._slice_actors.items():
            actor.GetMapper().SetLookupTable(lut)

    def enable_volume(self, enabled=True):
        """Toggle volume rendering."""
        if self._image_data is None:
            return

        if enabled and self._volume_actor is None:
            self._create_volume_actor()
        if self._volume_actor is not None:
            self._volume_actor.SetVisibility(enabled)
        self._volume_visible = enabled

    def set_volume_opacity(self, opacity=0.3):
        """Adjust volume rendering opacity."""
        if self._volume_actor is None:
            return
        prop = self._volume_actor.GetProperty()
        otf = vtkPiecewiseFunction()
        otf.AddPoint(0, 0.0)
        otf.AddPoint(50, 0.0)
        otf.AddPoint(100, opacity * 0.3)
        otf.AddPoint(200, opacity * 0.6)
        otf.AddPoint(255, opacity)
        prop.SetScalarOpacity(otf)

    def _create_volume_actor(self, colormap_name="Viridis"):
        """Create a volume actor with ray cast mapper."""
        mapper = vtkFixedPointVolumeRayCastMapper()
        mapper.SetInputData(self._image_data)
        mapper.SetSampleDistance(1.5)

        # Color transfer function
        ctf = vtkColorTransferFunction()
        cmap = _COLORMAPS.get(colormap_name, _COLORMAPS["Viridis"])
        for val_frac, r, g, b in cmap:
            ctf.AddRGBPoint(val_frac * 255, r, g, b)

        # Opacity transfer function — mostly transparent
        otf = vtkPiecewiseFunction()
        otf.AddPoint(0, 0.0)
        otf.AddPoint(50, 0.0)
        otf.AddPoint(100, 0.05)
        otf.AddPoint(200, 0.15)
        otf.AddPoint(255, 0.3)

        prop = vtkVolumeProperty()
        prop.SetColor(ctf)
        prop.SetScalarOpacity(otf)
        prop.SetInterpolationTypeToLinear()
        prop.ShadeOff()

        volume = vtkVolume()
        volume.SetMapper(mapper)
        volume.SetProperty(prop)

        self.renderer.AddVolume(volume)
        self._volume_actor = volume

    def _build_lut(self, colormap_name="Viridis"):
        """Build a vtkColorTransferFunction for slice coloring."""
        ctf = vtkColorTransferFunction()
        cmap = _COLORMAPS.get(colormap_name, _COLORMAPS["Viridis"])
        for val_frac, r, g, b in cmap:
            ctf.AddRGBPoint(val_frac * 255, r, g, b)
        return ctf

    def reset_camera(self):
        """Reset camera to show the full cube."""
        self.renderer.ResetCamera()
