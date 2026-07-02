import numpy as np
from skimage import measure

from multi_lif_analyzer_legacy import MultiSampleLifAnalyzer as LegacyMultiSampleLifAnalyzer


class ColocalizationAnalyzerMixin:
    def analyze_vesicle_localization(self, vesicles_binary, vesicles_corrected, cell_marker_channel, protein_corrected, sample_name):
        try:
            print(f"\n   Analyzing vesicle localization for: {sample_name}")

            vesicles_labels = measure.label(vesicles_binary)
            total_vesicles = int(np.max(vesicles_labels)) if np.max(vesicles_labels) > 0 else 0
            print(f"      Step 1: total vesicles detected = {total_vesicles}")

            vesicles_in_cells_mask = vesicles_binary & (cell_marker_channel > 0)
            vesicles_in_cells_labels = measure.label(vesicles_in_cells_mask)
            vesicles_in_cells_count = int(np.max(vesicles_in_cells_labels)) if np.max(vesicles_in_cells_labels) > 0 else 0
            print(f"      Step 2: vesicles in cells mask pixels = {int(np.sum(vesicles_in_cells_mask))}")
            print(f"      Step 2: vesicles in cells count = {vesicles_in_cells_count}")

            positive_protein = protein_corrected[protein_corrected > 0]
            if positive_protein.size > 0:
                protein_threshold = float(np.percentile(positive_protein, 50))
                print(f"      Step 3: protein threshold from positive pixels = {protein_threshold:.2f}")
                print(f"      Step 3: positive protein pixels = {positive_protein.size}")
            else:
                protein_threshold = 50.0
                print(f"      Step 3: no positive protein pixels found, using fallback threshold = {protein_threshold:.2f}")

            colocalized_mask = (vesicles_binary > 0) & (protein_corrected > protein_threshold)
            colocalized_vesicles_mask = np.logical_and(vesicles_binary, colocalized_mask)
            colocalized_labels = measure.label(colocalized_vesicles_mask)
            colocalized_count = len(np.unique(colocalized_labels)) - 1 if np.any(colocalized_vesicles_mask) else 0

            print(f"      Step 4: colocalized mask pixels = {int(np.sum(colocalized_mask))}")
            print(f"      Step 4: colocalized vesicles count = {colocalized_count}")

            if total_vesicles > 0:
                percent_in_cells = float(vesicles_in_cells_count / total_vesicles * 100)
            else:
                percent_in_cells = 0.0

            if vesicles_in_cells_count > 0:
                percent_colocalized = float(colocalized_count / vesicles_in_cells_count * 100)
            else:
                percent_colocalized = 0.0

            print(f"      Step 5: percent in cells = {percent_in_cells:.2f}%")
            print(f"      Step 5: percent colocalized = {percent_colocalized:.2f}%")

            return {
                'total_vesicles': total_vesicles,
                'vesicles_in_cells': vesicles_in_cells_count,
                'colocalized_vesicles': colocalized_count,
                'percent_in_cells': percent_in_cells,
                'percent_colocalized': percent_colocalized,
            }
        except Exception as e:
            print(f"   Error in analyze_vesicle_localization for {sample_name}: {e}")
            return {
                'total_vesicles': 0,
                'vesicles_in_cells': 0,
                'colocalized_vesicles': 0,
                'percent_in_cells': 0.0,
                'percent_colocalized': 0.0,
            }

    analyze_vesicles_in_cells = LegacyMultiSampleLifAnalyzer.analyze_vesicles_in_cells
    diagnose_colocalization = LegacyMultiSampleLifAnalyzer.diagnose_colocalization
