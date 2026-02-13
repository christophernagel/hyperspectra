"""
Constants and configuration for the hyperspectral viewer.
"""

# Atmospheric absorption bands (nm) - data unreliable in these regions
ATMOSPHERIC_BANDS = {
    'water_vapor_1': (1350, 1450),   # H2O absorption
    'water_vapor_2': (1800, 1950),   # H2O absorption
    'carbon_dioxide': (2500, 2600),  # CO2 absorption
    'oxygen': (760, 770),            # O2 A-band
}

# Numerical constants
DEFAULT_CACHE_SIZE = 50
MIN_REFLECTANCE = 0.001          # Below this is noise/bad data
PERCENTILE_LOW = 2               # For contrast stretching
PERCENTILE_HIGH = 98
ROBUST_PERCENTILE_LOW = 5        # For index clipping
ROBUST_PERCENTILE_HIGH = 95
LARGE_ARRAY_THRESHOLD = 1_000_000  # Sample above this size
SAMPLE_SIZE = 100_000            # Samples for percentile estimation

# Colorbar gradient definitions for Qt stylesheets
COLORBAR_GRADIENTS = {
    # Diverging colormaps
    'RdYlGn': 'stop:0 #a50026, stop:0.25 #f46d43, stop:0.5 #ffffbf, stop:0.75 #a6d96a, stop:1 #006837',
    'RdBu': 'stop:0 #67001f, stop:0.25 #d6604d, stop:0.5 #f7f7f7, stop:0.75 #4393c3, stop:1 #053061',
    'BrBG': 'stop:0 #543005, stop:0.25 #bf812d, stop:0.5 #f5f5f5, stop:0.75 #80cdc1, stop:1 #003c30',
    # Perceptually uniform
    'viridis': 'stop:0 #440154, stop:0.25 #3b528b, stop:0.5 #21918c, stop:0.75 #5ec962, stop:1 #fde725',
    'magma': 'stop:0 #000004, stop:0.25 #51127c, stop:0.5 #b73779, stop:0.75 #fc8961, stop:1 #fcfdbf',
    'plasma': 'stop:0 #0d0887, stop:0.25 #7e03a8, stop:0.5 #cc4778, stop:0.75 #f89540, stop:1 #f0f921',
    'inferno': 'stop:0 #000004, stop:0.25 #420a68, stop:0.5 #932667, stop:0.75 #dd513a, stop:1 #fca50a',
    'cividis': 'stop:0 #00224e, stop:0.25 #3d4e6e, stop:0.5 #7a7b78, stop:0.75 #b8a859, stop:1 #fee838',
    # Sequential - warm
    'YlOrBr': 'stop:0 #ffffe5, stop:0.25 #fec44f, stop:0.5 #ec7014, stop:0.75 #cc4c02, stop:1 #662506',
    'YlOrRd': 'stop:0 #ffffcc, stop:0.25 #feb24c, stop:0.5 #fd8d3c, stop:0.75 #e31a1c, stop:1 #800026',
    'OrRd': 'stop:0 #fff7ec, stop:0.25 #fdd49e, stop:0.5 #fc8d59, stop:0.75 #d7301f, stop:1 #7f0000',
    'Oranges': 'stop:0 #fff5eb, stop:0.25 #fdd0a2, stop:0.5 #fd8d3c, stop:0.75 #d94801, stop:1 #7f2704',
    'Reds': 'stop:0 #fff5f0, stop:0.25 #fcae91, stop:0.5 #fb6a4a, stop:0.75 #cb181d, stop:1 #67000d',
    'hot': 'stop:0 #0b0000, stop:0.25 #8b0000, stop:0.5 #ff4500, stop:0.75 #ffa500, stop:1 #ffff00',
    'copper': 'stop:0 #000000, stop:0.25 #4d2600, stop:0.5 #994c00, stop:0.75 #cc8033, stop:1 #ffc77f',
    # Sequential - cool
    'Blues': 'stop:0 #f7fbff, stop:0.25 #c6dbef, stop:0.5 #6baed6, stop:0.75 #2171b5, stop:1 #084594',
    'PuBu': 'stop:0 #fff7fb, stop:0.25 #d0d1e6, stop:0.5 #74a9cf, stop:0.75 #0570b0, stop:1 #023858',
    'Purples': 'stop:0 #fcfbfd, stop:0.25 #bcbddc, stop:0.5 #807dba, stop:0.75 #6a51a3, stop:1 #3f007d',
    'RdPu': 'stop:0 #fff7f3, stop:0.25 #fcc5c0, stop:0.5 #f768a1, stop:0.75 #ae017e, stop:1 #49006a',
    # Sequential - green
    'Greens': 'stop:0 #f7fcf5, stop:0.25 #a1d99b, stop:0.5 #41ab5d, stop:0.75 #006d2c, stop:1 #00441b',
    'YlGn': 'stop:0 #ffffe5, stop:0.25 #c2e699, stop:0.5 #78c679, stop:0.75 #238b45, stop:1 #004529',
    'BuGn': 'stop:0 #f7fcfd, stop:0.25 #b2e2e2, stop:0.5 #66c2a4, stop:0.75 #238b45, stop:1 #00441b',
}

# Index definitions with physically correct wavelengths
# References: USGS Spectral Library v7, Ninomiya (2003), Kühn (2004), Serrano (2002)
INDEX_DEFINITIONS = {
    # Vegetation Indices - Normalized Difference
    'NDVI': {'type': 'nd', 'b1': 850, 'b2': 670, 'cmap': 'RdYlGn'},
    'NDRE': {'type': 'nd', 'b1': 790, 'b2': 720, 'cmap': 'RdYlGn'},
    'NDWI': {'type': 'nd', 'b1': 560, 'b2': 860, 'cmap': 'RdBu'},
    'NDMI': {'type': 'nd', 'b1': 860, 'b2': 1650, 'cmap': 'BrBG'},

    # Clay Minerals - Continuum Removed (Al-OH absorption)
    'Clay (General)': {'type': 'continuum', 'feature': 2200, 'left': 2120, 'right': 2250, 'cmap': 'YlOrBr'},
    'Kaolinite': {'type': 'continuum', 'feature': 2205, 'left': 2120, 'right': 2250, 'cmap': 'Oranges'},
    'Alunite': {'type': 'continuum', 'feature': 2170, 'left': 2120, 'right': 2220, 'cmap': 'RdPu'},
    'Smectite': {'type': 'continuum', 'feature': 2200, 'left': 2120, 'right': 2290, 'cmap': 'YlOrRd'},
    'Illite': {'type': 'continuum', 'feature': 2195, 'left': 2120, 'right': 2250, 'cmap': 'OrRd'},

    # Carbonates - Continuum Removed (CO3 absorption)
    'Carbonate': {'type': 'continuum', 'feature': 2330, 'left': 2250, 'right': 2380, 'cmap': 'cividis'},
    'Calcite': {'type': 'continuum', 'feature': 2340, 'left': 2290, 'right': 2390, 'cmap': 'Blues'},
    'Dolomite': {'type': 'continuum', 'feature': 2320, 'left': 2260, 'right': 2380, 'cmap': 'PuBu'},
    'Chlorite': {'type': 'continuum', 'feature': 2330, 'left': 2250, 'right': 2380, 'cmap': 'Greens'},

    # Iron Oxides - Ratio (electronic transitions)
    'Iron Oxide': {'type': 'ratio', 'b1': 650, 'b2': 450, 'cmap': 'Reds'},
    'Ferric Iron': {'type': 'ratio', 'b1': 750, 'b2': 860, 'cmap': 'hot'},
    'Ferrous Iron': {'type': 'ratio', 'b1': 860, 'b2': 1000, 'cmap': 'copper'},

    # Hydrocarbons - C-H Absorption
    'Hydrocarbon': {'type': 'continuum', 'feature': 1730, 'left': 1660, 'right': 1780, 'cmap': 'Purples'},
    'HC 2310': {'type': 'continuum', 'feature': 2310, 'left': 2260, 'right': 2350, 'cmap': 'Purples'},
    'Methane': {'type': 'ratio', 'b1': 2260, 'b2': 2300, 'cmap': 'magma'},
    'Oil Slick': {'type': 'continuum', 'feature': 1730, 'left': 1680, 'right': 1760, 'cmap': 'inferno'},

    # Agriculture / Nitrogen
    'Protein': {'type': 'continuum', 'feature': 2170, 'left': 2100, 'right': 2230, 'cmap': 'Greens'},
    'Cellulose': {'type': 'continuum', 'feature': 2100, 'left': 2030, 'right': 2170, 'cmap': 'YlGn'},
    'Lignin': {'type': 'continuum', 'feature': 1680, 'left': 1620, 'right': 1740, 'cmap': 'BuGn'},
}

# Index metadata for UI labels and descriptions
INDEX_METADATA = {
    # Vegetation
    'NDVI': {'low': 'Bare soil / Water', 'high': 'Dense vegetation',
             'desc': 'Vegetation health (850/670nm). >0.3 = vegetation; >0.6 = dense.'},
    'NDRE': {'low': 'Stressed / Sparse', 'high': 'High chlorophyll',
             'desc': 'Red edge chlorophyll index. More sensitive than NDVI to stress.'},
    'NDWI': {'low': 'Dry / Land', 'high': 'Water bodies',
             'desc': 'Water detection. Positive = water; negative = land.'},
    'NDMI': {'low': 'Water stressed', 'high': 'High moisture',
             'desc': 'Moisture stress indicator. Tracks leaf water content.'},

    # Clay Minerals
    'Clay (General)': {'low': 'No clay', 'high': 'Strong Al-OH',
             'desc': '2200nm Al-OH absorption. Detects kaolinite, illite, smectite.'},
    'Kaolinite': {'low': 'No kaolinite', 'high': 'Strong kaolinite',
                  'desc': '2205nm Al-OH. Diagnostic doublet at 2165/2205nm. Argillic alteration.'},
    'Alunite': {'low': 'No alunite', 'high': 'Strong alunite',
                'desc': '2170nm Al-OH. Key for epithermal Au exploration. Cuprite signature mineral.'},
    'Smectite': {'low': 'No smectite', 'high': 'Strong smectite',
                 'desc': '2200nm broad Al-OH. Montmorillonite group. Wider absorption than kaolinite.'},
    'Illite': {'low': 'No illite', 'high': 'Strong illite',
               'desc': '2195nm Al-OH (shifted). Muscovite/sericite. Hydrothermal alteration indicator.'},

    # Carbonates
    'Carbonate': {'low': 'No carbonate', 'high': 'Strong CO3',
                  'desc': '2330nm CO3 vibration. General carbonate detection.'},
    'Calcite': {'low': 'No calcite', 'high': 'Strong calcite',
                'desc': '2340nm CO3. Limestone, marble. Longer wavelength than dolomite.'},
    'Dolomite': {'low': 'No dolomite', 'high': 'Strong dolomite',
                 'desc': '2320nm CO3. Mg shifts absorption ~20nm shorter than calcite.'},
    'Chlorite': {'low': 'No chlorite', 'high': 'Strong Mg-OH',
                 'desc': '2330nm Mg-OH. Chlorite/serpentine. Mafic alteration indicator.'},

    # Iron Oxides
    'Iron Oxide': {'low': 'Low iron', 'high': 'Iron oxide-rich',
                   'desc': 'Fe³⁺ charge transfer (650/450nm). Hematite, goethite, limonite.'},
    'Ferric Iron': {'low': 'Low Fe³⁺', 'high': 'High Fe³⁺',
                    'desc': 'Fe³⁺ crystal field (750/860nm). Red/yellow iron oxides.'},
    'Ferrous Iron': {'low': 'Low Fe²⁺', 'high': 'High Fe²⁺',
                     'desc': 'Fe²⁺ crystal field (860/1000nm). Pyroxene, olivine, chlorite.'},

    # Hydrocarbons
    'Hydrocarbon': {'low': 'No hydrocarbons', 'high': 'Strong C-H',
                'desc': '1730nm C-H 1st overtone. Primary oil/organic detection band.'},
    'HC 2310': {'low': 'No C-H', 'high': 'C-H combination',
                'desc': '2310nm C-H combination. Secondary hydrocarbon confirmation.'},
    'Methane': {'low': 'No CH4', 'high': 'CH4 absorption',
                'desc': '2300nm CH4 absorption. Natural gas seeps, emissions detection.'},
    'Oil Slick': {'low': 'No oil', 'high': 'Oil detected',
                  'desc': '1730nm optimized for marine oil slicks. Thickness indicator.'},

    # Agriculture
    'Protein': {'low': 'Low protein', 'high': 'High protein',
                'desc': '2170nm N-H. Leaf protein/nitrogen content. Crop health indicator.'},
    'Cellulose': {'low': 'Low cellulose', 'high': 'High cellulose',
                  'desc': '2100nm C-O. Cell wall content. High with low protein = stress.'},
    'Lignin': {'low': 'Low lignin', 'high': 'High lignin',
               'desc': '1680nm C-H. Woody material, senescent vegetation, dry litter.'},

    # Legacy indices (for custom calculations)
    'HC 1730': {'low': 'No hydrocarbons', 'high': 'C-H absorption',
                'desc': 'Legacy: C-H 1st overtone ratio (1680/1730nm).'},
    'HC (Light)': {'low': 'No light crude', 'high': 'Light HC',
                   'desc': 'Legacy: Light hydrocarbon transitions (900/1000nm).'},
    'Aromatic': {'low': 'Aliphatic', 'high': 'Aromatic-rich',
                 'desc': 'Aromatic vs aliphatic C-H ratio.'},
    'Chain_Length': {'low': 'Short chains', 'high': 'Long chains',
                     'desc': 'CH3/CH2 ratio indicates hydrocarbon chain length.'},
    'OH_Index': {'low': 'No O-H', 'high': 'O-H absorption',
                 'desc': 'Alcohol O-H overtone. In H2O absorption region.'},
    'Water': {'low': 'Dry', 'high': 'Wet/Water',
              'desc': 'Water absorption. In atmospheric H2O band.'},
    'NH_Index': {'low': 'No N-H', 'high': 'N-H absorption',
                 'desc': 'Primary amine N-H overtone bands.'},
    'Nitro': {'low': 'No nitro', 'high': 'Nitro absorption',
              'desc': 'N-O overtone. Nitro compound detection.'},
}

# Preset RGB composite definitions
COMPOSITE_PRESETS = {
    'True Color': {'r': 640, 'g': 550, 'b': 470},
    'CIR': {'r': 860, 'g': 650, 'b': 550},
    'SWIR Geology': {'r': 2200, 'g': 1650, 'b': 850},
    'Veg Stress': {'r': 750, 'g': 705, 'b': 550},
    'False NIR': {'r': 860, 'g': 750, 'b': 650},
    'Urban': {'r': 2200, 'g': 860, 'b': 650},
}

# Custom composite tips for common applications
COMPOSITE_TIPS = {
    'Vegetation Health': 'R:860 G:650 B:550 - CIR shows healthy vegetation as bright red',
    'Geology/Minerals': 'R:2200 G:1650 B:850 - SWIR highlights clay, carbonates, iron oxides',
    'Water Bodies': 'R:650 G:550 B:860 - Water appears dark, vegetation green',
    'Oil/Hydrocarbons': 'R:1730 G:1650 B:550 - Hydrocarbons show distinct spectral features',
    'Burn Scars': 'R:2190 G:860 B:650 - Burned areas appear red/magenta',
    'Snow/Ice': 'R:1650 G:860 B:550 - Snow bright in VNIR, dark in SWIR',
}

# Compound-based index suggestions
# Format: compound -> (index_name, b1_wl, b2_wl, description, index_type, colormap)
COMPOUND_SUGGESTIONS = {
    # Hydrocarbons
    'Petroleum (General)': ('HC 1730', 1680, 1730, 'C-H overtone absorption', 'ratio', 'Purples'),
    'Gasoline (Light HC)': ('Aromatic', 1670, 1730, 'Aromatic vs aliphatic', 'ratio', 'plasma'),
    'Diesel (Heavy HC)': ('Chain_Length', 1695, 1762, 'CH3/CH2 ratio', 'ratio', 'Purples'),
    'Alcohol': ('OH_Index', 1380, 1410, 'O-H overtone (in H2O band)', 'ratio', 'BrBG'),
    'Water': ('Water', 1380, 1450, 'H2O absorption (in atm. absorption)', 'ratio', 'RdBu'),
    # Minerals
    'Clay Minerals': ('Clay (General)', 2160, 2200, 'Al-OH absorption', 'ratio', 'YlOrBr'),
    'Carbonate': ('Carbonate', 2330, 2340, 'CO3 absorption', 'ratio', 'cividis'),
    'Iron Oxide': ('Iron Oxide', 650, 450, 'Fe charge transfer', 'ratio', 'Reds'),
    # Vegetation
    'Vegetation': ('NDVI', 860, 650, 'NIR vs Red', 'nd', 'RdYlGn'),
    # Organics
    'Primary Amine': ('NH_Index', 1486, 1526, 'NH2 overtone bands', 'ratio', 'BrBG'),
    'Aromatic': ('Aromatic', 1670, 1140, 'Aromatic C-H 1st/2nd OT ratio', 'ratio', 'plasma'),
    'Nitro Compounds': ('Nitro', 2180, 2300, 'N-O overtone', 'ratio', 'Purples'),
}

# =============================================================================
# SPECTRAL FEATURES FOR SMART RGB BUILDER
# Science-based wavelengths from USGS Spectral Library and literature
# Format: 'Display Name': (wavelength_nm, 'category', 'description')
# =============================================================================
SPECTRAL_FEATURES = {
    # Vegetation - based on chlorophyll absorption and cell structure
    'Chlorophyll Absorption (Blue)': (450, 'vegetation',
        'Peak chlorophyll-a absorption. Healthy vegetation absorbs strongly here.'),
    'Green Peak': (550, 'vegetation',
        'Chlorophyll reflection peak. The "green" we see in plants.'),
    'Red Absorption': (680, 'vegetation',
        'Second chlorophyll absorption maximum. Stressed plants reflect more here.'),
    'Red Edge Start': (705, 'vegetation',
        'Red edge inflection point. Sensitive to chlorophyll concentration and stress.'),
    'Red Edge Shoulder': (750, 'vegetation',
        'Transition to NIR plateau. Position shifts with plant health.'),
    'NIR Plateau': (860, 'vegetation',
        'High reflectance from leaf cell structure. Healthy vegetation is bright.'),
    'NIR Water (970nm)': (970, 'vegetation',
        'Leaf water absorption feature. Decreases with water stress.'),
    'Cellulose/Lignin': (1720, 'vegetation',
        'C-H bonds in plant biochemicals. High in dry/senescent vegetation.'),
    'Protein/Nitrogen': (2170, 'vegetation',
        'N-H absorption. Indicates leaf protein and nitrogen content.'),

    # Minerals - Clay/Phyllosilicates (Al-OH features)
    'Kaolinite (2160nm)': (2160, 'clay',
        'Al-OH bend+stretch. Diagnostic for kaolinite, halloysite.'),
    'Clay Al-OH (2200nm)': (2200, 'clay',
        'Primary Al-OH absorption. Present in most clay minerals.'),
    'Smectite/Montmorillonite': (2210, 'clay',
        'Al-OH in expandable clays. Slightly shifted from kaolinite.'),
    'Illite/Muscovite': (2350, 'clay',
        'Al-OH in micas. Distinguishes from kaolinite group.'),

    # Minerals - Carbonates (CO3 features)
    'Carbonate (2330nm)': (2330, 'carbonate',
        'C-O stretch overtone. Strong in calcite, dolomite, limestone.'),
    'Calcite': (2340, 'carbonate',
        'Calcite-specific CO3 absorption position.'),
    'Dolomite': (2320, 'carbonate',
        'Mg-carbonate CO3 position. Shifted from calcite.'),

    # Minerals - Iron (electronic transitions)
    'Ferric Iron (650nm)': (650, 'iron',
        'Fe3+ charge transfer. Hematite, goethite appear red/orange.'),
    'Iron Absorption (900nm)': (900, 'iron',
        'Fe crystal field absorption. Present in many iron-bearing minerals.'),
    'Ferrous Iron (1000nm)': (1000, 'iron',
        'Fe2+ absorption. Pyroxene, olivine, some clays.'),
    'Goethite': (480, 'iron',
        'Fe3+ in goethite. Distinguishes from hematite.'),
    'Hematite': (550, 'iron',
        'Fe3+ in hematite. Gives characteristic red color.'),

    # Minerals - Mg-OH (Chlorite, Serpentine)
    'Chlorite (2250nm)': (2250, 'mgoh',
        'Mg-OH absorption. Chlorite, serpentine group minerals.'),
    'Serpentine (2320nm)': (2320, 'mgoh',
        'Mg-OH in serpentine. Alteration indicator.'),

    # Hydrocarbons - C-H absorption features
    'HC C-H Primary (1730nm)': (1730, 'hydrocarbon',
        'C-H 1st overtone. Primary hydrocarbon detection wavelength.'),
    'HC C-H Combo (2310nm)': (2310, 'hydrocarbon',
        'C-H combination band. Confirms hydrocarbon presence.'),
    'Methyl CH3 (1695nm)': (1695, 'hydrocarbon',
        'CH3 asymmetric stretch. Higher in light hydrocarbons.'),
    'Methylene CH2 (1762nm)': (1762, 'hydrocarbon',
        'CH2 symmetric stretch. Higher in heavy/long-chain HCs.'),
    'Aromatic C-H': (1670, 'hydrocarbon',
        'Aromatic ring C-H. Gasoline, BTEX compounds.'),

    # Water/Moisture features
    'Water 1400nm': (1400, 'water',
        'O-H stretch overtone. Liquid water and hydroxyl minerals.'),
    'Water 1900nm': (1900, 'water',
        'H2O combination band. Very strong - often saturated.'),
    'Free Water': (970, 'water',
        'O-H overtone. Surface/canopy water content.'),

    # Standard visible for true color
    'Blue (470nm)': (470, 'visible',
        'Standard blue band. Water penetration, atmospheric scatter.'),
    'Green (550nm)': (550, 'visible',
        'Standard green band. Vegetation reflectance peak.'),
    'Red (640nm)': (640, 'visible',
        'Standard red band. Good contrast, vegetation absorption.'),
    'Deep Blue (420nm)': (420, 'visible',
        'Coastal/aerosol band. Atmospheric correction, shallow water.'),
}

# Reference spectral signatures for peak matching
# Format: 'Material': [(wavelength, relative_depth, 'feature'), ...]
# relative_depth: 0-1 scale, 1 = deepest absorption
REFERENCE_SIGNATURES = {
    'Kaolinite': [
        (1400, 0.4, 'OH stretch'),
        (1900, 0.5, 'H2O'),
        (2160, 0.9, 'Al-OH doublet'),
        (2200, 1.0, 'Al-OH primary'),
    ],
    'Montmorillonite': [
        (1400, 0.6, 'OH stretch'),
        (1900, 0.9, 'H2O interlayer'),
        (2210, 1.0, 'Al-OH'),
    ],
    'Illite': [
        (1400, 0.5, 'OH'),
        (1900, 0.4, 'H2O'),
        (2200, 0.8, 'Al-OH'),
        (2350, 1.0, 'Al-OH mica'),
    ],
    'Calcite': [
        (1900, 0.3, 'H2O'),
        (2340, 1.0, 'CO3'),
        (2500, 0.6, 'CO3 overtone'),
    ],
    'Dolomite': [
        (2320, 1.0, 'CO3 Mg-shifted'),
        (2500, 0.5, 'CO3 overtone'),
    ],
    'Goethite': [
        (480, 0.7, 'Fe3+ CT'),
        (650, 0.5, 'Fe3+ CT'),
        (900, 1.0, 'Fe3+ crystal field'),
    ],
    'Hematite': [
        (550, 0.6, 'Fe3+ CT'),
        (650, 0.4, 'Fe3+ CT'),
        (860, 1.0, 'Fe3+ crystal field'),
    ],
    'Chlorite': [
        (2250, 1.0, 'Mg-OH'),
        (2350, 0.7, 'Mg-OH'),
    ],
    'Healthy Vegetation': [
        (450, 0.8, 'Chlorophyll blue'),
        (680, 1.0, 'Chlorophyll red'),
        (1400, 0.3, 'Leaf water'),
        (1900, 0.4, 'Leaf water'),
    ],
    'Stressed Vegetation': [
        (450, 0.5, 'Reduced chlorophyll'),
        (680, 0.6, 'Reduced chlorophyll'),
        (1720, 0.7, 'Dry matter'),
        (2100, 0.5, 'Cellulose'),
    ],
    'Dry Vegetation': [
        (1720, 1.0, 'Lignin/cellulose'),
        (2100, 0.8, 'Cellulose'),
        (2300, 0.5, 'Lignin'),
    ],
    'Crude Oil': [
        (1730, 1.0, 'C-H overtone'),
        (2310, 0.8, 'C-H combo'),
    ],
    'Asphalt': [
        (1730, 0.9, 'C-H overtone'),
        (2310, 1.0, 'C-H combo'),
        (1762, 0.7, 'CH2'),
    ],
    'Gasoline/Light HC': [
        (1670, 1.0, 'Aromatic C-H'),
        (1695, 0.8, 'CH3'),
    ],
}

# =============================================================================
# INDOOR MATERIAL SPECTRAL FEATURES (for HyperspecI and close-range imaging)
#
# Wavelength positions verified against published NIR/SWIR spectroscopy literature:
#   - Wilson et al. (2015) "Review of short-wave infrared spectroscopy and imaging
#     methods for biological tissue characterization" J. Biomed. Opt. 20(3):030901
#   - Weyer & Lo (2002) "Spectra-Structure Correlations in the Near-Infrared"
#     in Handbook of Vibrational Spectroscopy
#
# IMPORTANT: Relative absorption depths (0-1 scale) are APPROXIMATE placeholders.
# For quantitative material matching, replace with measured spectra from:
#   - USGS Spectral Library v7 Chapter A (manmade materials): https://doi.org/10.5066/F7RR1WDJ
#   - ECOSTRESS/ASTER Spectral Library: https://speclib.jpl.nasa.gov/
#
# HyperspecI spectral ranges:
#   D1: 400-1000nm (61 channels) - VNIR only, limited SWIR features
#   D2: 400-1700nm (131 channels) - Includes first overtone C-H/N-H/O-H bands
# =============================================================================

INDOOR_MATERIAL_SIGNATURES = {
    # Textiles and Fabrics
    'Cotton (White)': [
        (970, 0.5, 'Cellulose O-H'),
        (1200, 0.6, 'C-H 2nd overtone'),
        (1490, 0.8, 'Cellulose O-H 1st OT'),
    ],
    'Cotton (Colored)': [
        (970, 0.4, 'Cellulose O-H'),
        (1200, 0.5, 'C-H 2nd overtone'),
        (1490, 0.7, 'Cellulose O-H 1st OT'),
    ],
    'Polyester': [
        (1130, 0.7, 'C-H 2nd OT'),
        (1410, 0.5, 'O-H impurity'),
        (1660, 0.9, 'C-H 1st OT aromatic'),
    ],
    'Nylon': [
        (1030, 0.5, 'N-H 2nd OT'),
        (1500, 1.0, 'N-H 1st OT'),
        (1200, 0.6, 'C-H 2nd OT'),
    ],
    'Wool': [
        (1020, 0.6, 'N-H 2nd OT (protein)'),
        (1190, 0.5, 'C-H 2nd OT'),
        (1510, 0.9, 'N-H 1st OT'),
    ],
    'Silk': [
        (1020, 0.7, 'N-H 2nd OT (protein)'),
        (1510, 1.0, 'N-H 1st OT'),
    ],

    # Plastics
    'Polyethylene (PE)': [
        (1210, 0.8, 'CH2 2nd OT'),
        (1400, 0.5, 'CH2 combo'),
        # NOTE: Primary 1730nm peak is outside D2 range (400-1700nm)
    ],
    'Polypropylene (PP)': [
        (1190, 0.7, 'CH3 2nd OT'),
        (1380, 0.6, 'CH3 combo'),
        # NOTE: 1720nm near D2 edge - may be partially captured
    ],
    'PVC': [
        (1180, 0.6, 'C-H 2nd OT'),
        (1420, 0.5, 'CH2 combo'),
        (1710, 0.9, 'C-H 1st OT'),
    ],
    'Polystyrene': [
        (1140, 0.7, 'Aromatic C-H 2nd OT'),
        (1670, 1.0, 'Aromatic C-H 1st OT'),
    ],
    'Acrylic (PMMA)': [
        (1180, 0.6, 'C-H 2nd OT'),
        (1430, 0.5, 'O-H impurity'),
        (1690, 0.9, 'C-H 1st OT'),
    ],
    'ABS Plastic': [
        (1150, 0.6, 'C-H 2nd OT'),
        (1670, 0.9, 'Aromatic C-H 1st OT'),
        # NOTE: 1730nm aliphatic peak is outside D2 range
    ],

    # Food and Organic Materials
    'Fresh Fruit': [
        (680, 0.4, 'Chlorophyll'),
        (970, 0.8, 'Water O-H'),
        (1450, 1.0, 'Water O-H 1st OT'),
    ],
    'Dried Fruit': [
        (980, 0.5, 'Residual water'),
        (1200, 0.7, 'Sugar C-H'),
        (1450, 0.6, 'Residual water'),
    ],
    'Bread/Baked Goods': [
        (980, 0.5, 'Moisture'),
        (1200, 0.8, 'Starch C-H'),
        (1450, 0.6, 'Moisture'),
        (1510, 0.4, 'Protein N-H'),
    ],
    'Meat (Raw)': [
        (760, 0.5, 'Myoglobin'),
        (970, 0.9, 'Water'),
        (1450, 1.0, 'Water O-H'),
        (1510, 0.6, 'Protein N-H'),
    ],
    'Meat (Cooked)': [
        (970, 0.6, 'Reduced water'),
        (1210, 0.5, 'Fat C-H 2nd OT'),
        (1450, 0.7, 'Water'),
        (1500, 0.5, 'Protein N-H'),
        # NOTE: 1730nm fat peak outside D2 range, using 1210nm proxy
    ],
    'Cheese': [
        (970, 0.7, 'Moisture'),
        (1210, 0.6, 'Fat C-H 2nd OT'),
        (1450, 0.8, 'Moisture'),
        (1500, 0.5, 'Protein N-H'),
        # NOTE: 1730nm fat peak outside D2 range, using 1210nm proxy
    ],
    'Leafy Vegetables': [
        (550, 0.3, 'Chlorophyll reflectance'),
        (680, 0.9, 'Chlorophyll absorption'),
        (970, 0.8, 'Leaf water'),
        (1450, 1.0, 'Leaf water'),
    ],

    # Wood and Paper
    'Wood (Light)': [
        (980, 0.4, 'Moisture'),
        (1200, 0.6, 'Cellulose C-H'),
        (1450, 0.7, 'Moisture'),
        (1490, 0.9, 'Cellulose O-H'),
        (1680, 0.5, 'Lignin'),
    ],
    'Wood (Dark)': [
        (980, 0.3, 'Moisture'),
        (1200, 0.5, 'Cellulose C-H'),
        (1490, 0.8, 'Cellulose O-H'),
        (1680, 0.7, 'Lignin (higher)'),
    ],
    'Paper (White)': [
        (970, 0.3, 'Low moisture'),
        (1200, 0.7, 'Cellulose C-H'),
        (1490, 1.0, 'Cellulose O-H'),
    ],
    'Cardboard': [
        (980, 0.4, 'Moisture'),
        (1200, 0.6, 'Cellulose C-H'),
        (1490, 0.9, 'Cellulose O-H'),
        (1680, 0.5, 'Lignin'),
    ],

    # Metals and Coatings
    'Painted Metal (White)': [
        (1180, 0.4, 'Paint C-H 2nd OT'),
        (1430, 0.3, 'Paint O-H'),
        # NOTE: 1730nm peak outside D2 range
    ],
    'Painted Metal (Colored)': [
        (1180, 0.5, 'Paint C-H 2nd OT'),
        (1670, 0.4, 'Aromatic if present'),
    ],
    'Anodized Aluminum': [
        # Generally flat spectrum with dye absorption
        (550, 0.5, 'Dye absorption varies'),
    ],
    'Stainless Steel': [
        # Very flat, low reflectance spectrum
    ],

    # Ceramics and Glass
    'Ceramic (Glazed)': [
        (1400, 0.4, 'Si-OH'),
        # NOTE: 1900nm H2O feature outside D2 range
    ],
    'Glass (Clear)': [
        (1400, 0.3, 'Si-OH'),
        # Mostly flat spectrum - weak features
    ],
    'Porcelain': [
        (1400, 0.5, 'Si-OH'),
    ],

    # Skin and Hair
    'Human Skin': [
        (550, 0.4, 'Hemoglobin'),
        (580, 0.5, 'Melanin'),
        (970, 0.6, 'Water'),
        (1200, 0.4, 'Lipid C-H'),
        (1450, 0.8, 'Water O-H'),
    ],
    'Hair (Dark)': [
        (1180, 0.5, 'Keratin C-H'),
        (1510, 0.9, 'Keratin N-H'),
    ],
    'Hair (Light)': [
        (1180, 0.4, 'Keratin C-H'),
        (1510, 0.8, 'Keratin N-H'),
    ],
}

# =============================================================================
# INDOOR SPECTRAL INDICES
#
# These indices are designed for indoor/close-range hyperspectral data.
# Each index specifies which HyperspecI database(s) it works with:
#   - 'range': (min_wl, max_wl) in nm - the wavelengths the index requires
#   - D1 (400-1000nm) indices work on both databases
#   - D2-only indices require the 400-1700nm extended range
#
# Key absorption bands (from Wilson et al. 2015, PMC4370890):
#   Water:    970nm (2nd OT), 1430nm (1st OT), 1940nm (combo)
#   Lipid:    920nm (2nd OT), 1210nm (2nd OT), 1730nm (1st OT - 80x stronger)
#   Protein:  1200nm (2nd OT), 1500nm (combo), 1690nm (1st OT CH3)
#   Collagen: 1200nm, 1500nm, 1690nm, 1725nm
#
# NOTE: 1730nm is OUTSIDE D2 range (400-1700nm). Use 1210nm (2nd overtone) or
#       1690nm (edge of range) as proxies for lipid detection.
# =============================================================================

INDOOR_INDEX_DEFINITIONS = {
    # -------------------------------------------------------------------------
    # D1-compatible indices (400-1000nm) - work on both D1 and D2
    # -------------------------------------------------------------------------

    # Chlorophyll (plants, leafy items) - standard red edge
    'Chlorophyll': {
        'type': 'nd', 'b1': 750, 'b2': 680, 'cmap': 'RdYlGn',
        'range': (680, 750),
    },

    # Moisture - weak 2nd overtone O-H at 970nm (D1 edge band - may be noisy)
    'Moisture (VNIR)': {
        'type': 'ratio', 'b1': 970, 'b2': 900, 'cmap': 'Blues',
        'range': (900, 970),
    },

    # -------------------------------------------------------------------------
    # D2-only indices (require 400-1700nm coverage)
    # -------------------------------------------------------------------------

    # Moisture - stronger 1st overtone O-H at 1430nm
    'Moisture (SWIR)': {
        'type': 'continuum', 'feature': 1430, 'left': 1350, 'right': 1500, 'cmap': 'Blues',
        'range': (1350, 1500),
    },

    # Protein content - N-H combination band at 1500nm (collagen peak)
    'Protein': {
        'type': 'continuum', 'feature': 1500, 'left': 1450, 'right': 1550, 'cmap': 'Purples',
        'range': (1450, 1550),
    },

    # Lipid - using 1210nm 2nd overtone (within D2 range, 12x weaker than 1730nm)
    # NOTE: Primary 1730nm peak is outside D2 range!
    'Lipid (2nd OT)': {
        'type': 'continuum', 'feature': 1210, 'left': 1150, 'right': 1260, 'cmap': 'YlOrBr',
        'range': (1150, 1260),
    },

    # Lipid - using 1690nm edge (CH3 1st overtone, at D2 limit)
    # This may be partially captured but accuracy is uncertain
    'Lipid (Edge)': {
        'type': 'ratio', 'b1': 1690, 'b2': 1600, 'cmap': 'YlOrBr',
        'range': (1600, 1690),
    },

    # Cellulose - O-H combination bands
    'Cellulose': {
        'type': 'continuum', 'feature': 1490, 'left': 1420, 'right': 1550, 'cmap': 'BuGn',
        'range': (1420, 1550),
    },

    # Synthetic polymer detection - C-H vs N-H ratio
    # Higher values = more aliphatic C-H (synthetic), lower = more N-H (protein/natural)
    'Polymer': {
        'type': 'ratio', 'b1': 1210, 'b2': 1500, 'cmap': 'plasma',
        'range': (1210, 1500),
    },

    # Aromatic compounds - aromatic C-H 1st overtone at 1670nm
    'Aromatic': {
        'type': 'continuum', 'feature': 1670, 'left': 1620, 'right': 1700, 'cmap': 'inferno',
        'range': (1620, 1700),
    },
}

INDOOR_INDEX_METADATA = {
    'Chlorophyll': {
        'low': 'No chlorophyll', 'high': 'Chlorophyll present',
        'desc': 'Red edge NDVI-like index (750/680nm). Works on D1 and D2.',
        'database': 'D1, D2',
    },
    'Moisture (VNIR)': {
        'low': 'Dry', 'high': 'Wet/Moist',
        'desc': 'Weak O-H 2nd overtone at 970nm. Edge of D1 range - may be noisy.',
        'database': 'D1, D2',
    },
    'Moisture (SWIR)': {
        'low': 'Dry', 'high': 'Wet/Moist',
        'desc': 'Strong O-H 1st overtone at 1430nm. 60x stronger than 970nm peak.',
        'database': 'D2 only',
    },
    'Protein': {
        'low': 'Low protein', 'high': 'High protein',
        'desc': 'N-H combination band at 1500nm. High in meat, cheese, wool, silk, collagen.',
        'database': 'D2 only',
    },
    'Lipid (2nd OT)': {
        'low': 'Low fat', 'high': 'High fat',
        'desc': 'C-H 2nd overtone at 1210nm. 12x weaker than 1730nm but within D2 range.',
        'database': 'D2 only',
    },
    'Lipid (Edge)': {
        'low': 'Low fat', 'high': 'High fat',
        'desc': 'CH3 1st overtone at 1690nm. At D2 spectral limit - accuracy uncertain.',
        'database': 'D2 only (edge)',
    },
    'Cellulose': {
        'low': 'Low cellulose', 'high': 'High cellulose',
        'desc': 'O-H combination at 1490nm. Paper, cotton, wood, plant cell walls.',
        'database': 'D2 only',
    },
    'Polymer': {
        'low': 'Natural (protein-rich)', 'high': 'Synthetic (C-H rich)',
        'desc': 'Ratio of C-H (1210nm) to N-H (1500nm). Distinguishes plastic from organic.',
        'database': 'D2 only',
    },
    'Aromatic': {
        'low': 'Aliphatic', 'high': 'Aromatic-rich',
        'desc': 'Aromatic C-H 1st overtone at 1670nm. Polystyrene, paint, dyes.',
        'database': 'D2 only',
    },
}
