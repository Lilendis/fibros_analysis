import numpy as np
from skimage import measure, morphology

from multi_lif_analyzer_legacy import MultiSampleLifAnalyzer as LegacyMultiSampleLifAnalyzer


class SegmenterMixin:
    segment_vesicles = LegacyMultiSampleLifAnalyzer.segment_vesicles
    segment_vesicles_with_criteria = LegacyMultiSampleLifAnalyzer.segment_vesicles_with_criteria
    segment_vesicles_from_mask = LegacyMultiSampleLifAnalyzer.segment_vesicles_from_mask
    adaptive_grid_segmentation = LegacyMultiSampleLifAnalyzer.adaptive_grid_segmentation
    smart_grid_segmentation = LegacyMultiSampleLifAnalyzer.smart_grid_segmentation
    watershed_intensity_segmentation = LegacyMultiSampleLifAnalyzer.watershed_intensity_segmentation
    uniform_grid_segmentation_improved = LegacyMultiSampleLifAnalyzer.uniform_grid_segmentation_improved
    finalize_vesicles = LegacyMultiSampleLifAnalyzer.finalize_vesicles
    segment_around_bright_centers = LegacyMultiSampleLifAnalyzer.segment_around_bright_centers
    uniform_grid_segmentation = LegacyMultiSampleLifAnalyzer.uniform_grid_segmentation
    intensity_based_segmentation = LegacyMultiSampleLifAnalyzer.intensity_based_segmentation

    def segment_entire_red_area(self, full_red_area, vesicles_channel):
        """Segments the entire red area into vesicles with iterative splitting of large clusters."""
        try:
            print("      Segmenting entire red area...")

            if full_red_area is None or vesicles_channel is None:
                print("      Warning: empty input data")
                return np.zeros_like(full_red_area, dtype=np.uint8)

            if not np.any(full_red_area):
                print("      Warning: no red area for segmentation")
                return np.zeros_like(full_red_area, dtype=np.uint8)

            red_labels = measure.label(full_red_area)
            red_regions = measure.regionprops(red_labels, intensity_image=vesicles_channel)

            target_vesicle_size = (self.min_vesicle_size + self.max_vesicle_size) // 2
            print(f"      Target vesicle size: {target_vesicle_size}px")

            final_vesicles_mask = np.zeros_like(full_red_area, dtype=bool)
            total_clusters = 0
            total_created_vesicles = 0

            for i, region in enumerate(red_regions):
                try:
                    region_area = region.area
                    region_mask = red_labels == region.label
                    perimeter = region.perimeter
                    circularity = 4 * np.pi * region_area / (perimeter ** 2) if perimeter > 0 else 0.0

                    print(f"      Region {i + 1}: area={region_area}px, circularity={circularity:.3f}")

                    if region_area < self.min_vesicle_size:
                        continue

                    is_cluster = False
                    if region_area > self.max_vesicle_size * 1.5:
                        is_cluster = True
                    if circularity < 0.5 and region_area > self.max_vesicle_size:
                        is_cluster = True

                    if not is_cluster and region_area <= self.max_vesicle_size:
                        final_vesicles_mask[region_mask] = True
                        total_created_vesicles += 1
                        print("         Keeping as single vesicle")
                        continue

                    if not self.split_large_clusters:
                        final_vesicles_mask[region_mask] = True
                        total_created_vesicles += 1
                        print("         Large cluster splitting disabled, keeping region as-is")
                        continue

                    total_clusters += 1
                    current_mask = region_mask.copy()
                    current_area = int(np.sum(current_mask))
                    best_mask = np.zeros_like(current_mask, dtype=bool)
                    max_iterations = 3
                    iteration = 0

                    print(f"         Cluster detected: area={current_area}px, circularity={circularity:.3f}")

                    while iteration < max_iterations and current_area > self.max_vesicle_size * 1.2:
                        iteration += 1
                        print(f"         Split iteration {iteration}: current area={current_area}px")

                        segmented_mask = self.optimal_cluster_segmentation(
                            current_mask,
                            current_area,
                            target_vesicle_size,
                            vesicles_channel,
                        )

                        if segmented_mask is None or not np.any(segmented_mask):
                            print("            No segmentation result, stopping iterations")
                            break

                        best_mask = segmented_mask.copy()
                        labels_after = measure.label(segmented_mask)
                        regions_after = measure.regionprops(labels_after)
                        areas_after = [r.area for r in regions_after]
                        large_regions = [area for area in areas_after if area > self.max_vesicle_size * 1.2]

                        print(f"            Obtained vesicles: {len(regions_after)}")
                        print(f"            Coverage after split: {np.sum(segmented_mask) / current_area * 100:.1f}%")

                        if not large_regions:
                            print("            All large regions successfully split")
                            break

                        largest_area = max(large_regions)
                        current_mask = segmented_mask
                        current_area = largest_area
                        print(f"            Remaining oversized region detected: {largest_area}px")

                    cluster_result = best_mask if np.any(best_mask) else current_mask
                    cluster_count = np.max(measure.label(cluster_result)) if np.any(cluster_result) else 0
                    final_vesicles_mask[cluster_result] = True
                    total_created_vesicles += cluster_count

                    print(f"         Total split iterations: {iteration}")
                    print(f"         Created vesicles from region: {cluster_count}")

                except Exception as region_error:
                    print(f"         Error while processing region {i + 1}: {region_error}")
                    continue

            try:
                final_cleaned = morphology.remove_small_objects(
                    final_vesicles_mask,
                    min_size=self.min_vesicle_size,
                )

                final_pixels = np.sum(final_cleaned)
                original_pixels = np.sum(full_red_area)
                coverage = final_pixels / original_pixels * 100 if original_pixels > 0 else 0

                print("      Final segmentation summary:")
                print(f"         Regions processed: {len(red_regions)}")
                print(f"         Clusters processed: {total_clusters}")
                print(f"         Total created vesicles: {total_created_vesicles}")
                print(f"         Coverage after segmentation: {coverage:.1f}%")

                return final_cleaned.astype(np.uint8) * 255
            except Exception as postprocess_error:
                print(f"      Post-processing error: {postprocess_error}")
                return final_vesicles_mask.astype(np.uint8) * 255

        except Exception as main_error:
            print(f"      Critical error in segment_entire_red_area: {main_error}")
            if full_red_area is not None:
                return np.zeros_like(full_red_area, dtype=np.uint8)
            return np.array([], dtype=np.uint8)

    def optimal_cluster_segmentation(self, cluster_mask, cluster_area, target_size, vesicles_channel):
        """Segments a cluster and falls back across methods if coverage is poor."""
        try:
            optimal_count = max(2, int(cluster_area / target_size))

            coords = np.where(cluster_mask)
            if len(coords[0]) == 0 or len(coords[1]) == 0:
                print("            Warning: empty cluster")
                return np.zeros_like(cluster_mask, dtype=bool)

            y_min, y_max = np.min(coords[0]), np.max(coords[0])
            x_min, x_max = np.min(coords[1]), np.max(coords[1])
            height, width = y_max - y_min + 1, x_max - x_min + 1

            print(f"            Source cluster size: {cluster_area}px")
            print(f"            Bounding box: {width}x{height}")
            print(f"            Target vesicle count: {optimal_count}")

            methods = [
                ("watershed", lambda: self.watershed_intensity_segmentation(cluster_mask, vesicles_channel, optimal_count)),
                (
                    "adaptive_grid",
                    lambda: self.adaptive_grid_segmentation(
                        cluster_mask,
                        vesicles_channel,
                        y_min,
                        y_max,
                        x_min,
                        x_max,
                        optimal_count,
                        target_size,
                    ),
                ),
                (
                    "uniform_grid",
                    lambda: self.uniform_grid_segmentation_improved(
                        cluster_mask,
                        cluster_area,
                        target_size,
                        y_min,
                        y_max,
                        x_min,
                        x_max,
                    ),
                ),
            ]

            result_mask = np.zeros_like(cluster_mask, dtype=bool)

            for method_name, method_fn in methods:
                print(f"            Trying method: {method_name}")
                candidate_mask = method_fn()
                if candidate_mask is None:
                    candidate_mask = np.zeros_like(cluster_mask, dtype=bool)

                result_area = int(np.sum(candidate_mask))
                coverage = result_area / cluster_area if cluster_area > 0 else 0.0
                candidate_count = int(np.max(measure.label(candidate_mask))) if np.any(candidate_mask) else 0

                print(f"               Result area: {result_area}px")
                print(f"               Coverage after segmentation: {coverage * 100:.1f}%")
                print(f"               Obtained vesicles: {candidate_count}")

                result_mask = candidate_mask
                if coverage >= 0.6:
                    print(f"               Method accepted: {method_name}")
                    break

                print(f"               Coverage below threshold, trying next method")

            if result_mask is not None and np.sum(result_mask) > 0:
                final_vesicles = self.finalize_vesicles(result_mask, target_size)
                final_coverage = np.sum(final_vesicles) / cluster_area * 100 if cluster_area > 0 else 0
                final_count = np.max(measure.label(final_vesicles)) if np.any(final_vesicles) else 0

                print(f"            Final vesicles count: {final_count}")
                print(f"            Final coverage: {final_coverage:.1f}%")
                return final_vesicles

            print("            Warning: failed to segment cluster")
            return np.zeros_like(cluster_mask, dtype=bool)

        except Exception as e:
            print(f"            Error in optimal_cluster_segmentation: {e}")
            return self.uniform_grid_segmentation(cluster_mask, cluster_area, target_size)
