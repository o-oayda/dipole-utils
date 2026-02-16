fn_dict = {
    'quaia': {
        'high': {
            'catalogue': 'quaia_G20.5.fits',
            'selection_function': 'selection_function_NSIDE64_G20.5.fits'
        },
        'low': {
            'catalogue': 'quaia_G20.0.fits',
            'selection_function': 'selection_function_NSIDE64_G20.0.fits'
        },
        'high_v2': {
            'catalogue': 'quaia_v2_G20.5.fits',
            'selection_function': 'selection_function_v2_NSIDE64_G20.5.fits'
        },
        'low_v2': {
            'catalogue': 'quaia_v2_G20.0.fits',
            'selection_function': 'selection_function_v2_NSIDE64_G20.0.fits'
        }
    },
    'catwise': {
        '2021': 'catwise_agns_masked_final_w1lt16p5_alpha.fits',
        # '2025': 'w12_0.5_corrected/catwise_agns_corr.fits',
        # '2025-deep': f'{header}/NICE-DRIVE/research_data/surveys/catwise_raw/w12_0p5_w1_17p0_corrected/catwise_agns_corr.fits'
    },
    'milliquas': 'milliquas.fits',
    'vlass': 'VLASS_processed.csv',
    'racs': {
        'low1': 'AS110_Derived_Catalogue_racs_dr1_sources_galacticcut_v2021_08_v02_5725.csv',
        'low2': 'RACS-low2_sources.fits',
        'low2-patch': 'RACS-low2_sources_patched.fits',
        'low2-45as': 'RACS-low2_sources_45arcsec.fits',
        'low2-45as-patch': 'RACS-low2_sources_45arcsec_patched.fits',
        'low2-25as-patch': 'RACS-low2_sources_25arcsec_patched.fits',
        'low3': 'RACS-low3_sources.fits',
        'low3-scaled': 'RACS-low3_sources_scaled.fits',
        'mid1': 'AS110_Derived_Catalogue_racs_mid_sources_v01_15372.csv',
        'mid1-25as': 'RACS-mid1_25arcsec_sources.csv',
        'high1': 'RACS-high_sources_21-01-25.fits'
    },
    'catnorth': 'catnorth_qso_cand.fits',
    'nvss': 'Full_NVSS_combined_named.dat',
    'erass': 'eRASS1_Main.v1.1.fits',
    'qucats': 'QuCatS_Nakazono_and_Valenca_2024.csv',
    'auger': 'events_a8_stripped.csv',
    'gleam': {
        'egc': 'gleamegc.txt',
        'x-dr2': 'VIII_113_catalog2.dat.gz.fits'
    }
}
