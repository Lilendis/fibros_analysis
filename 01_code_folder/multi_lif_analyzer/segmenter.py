import cv2
import numpy as np
from scipy import ndimage
from skimage import filters, measure, morphology

from multi_lif_analyzer_legacy import MultiSampleLifAnalyzer as LegacyMultiSampleLifAnalyzer


class SegmenterMixin:
    segment_vesicles_from_mask = LegacyMultiSampleLifAnalyzer.segment_vesicles_from_mask
    adaptive_grid_segmentation = LegacyMultiSampleLifAnalyzer.adaptive_grid_segmentation
    smart_grid_segmentation = LegacyMultiSampleLifAnalyzer.smart_grid_segmentation
    watershed_intensity_segmentation = LegacyMultiSampleLifAnalyzer.watershed_intensity_segmentation
    uniform_grid_segmentation_improved = LegacyMultiSampleLifAnalyzer.uniform_grid_segmentation_improved
    finalize_vesicles = LegacyMultiSampleLifAnalyzer.finalize_vesicles
    segment_around_bright_centers = LegacyMultiSampleLifAnalyzer.segment_around_bright_centers
    uniform_grid_segmentation = LegacyMultiSampleLifAnalyzer.uniform_grid_segmentation
    intensity_based_segmentation = LegacyMultiSampleLifAnalyzer.intensity_based_segmentation

    def segment_vesicles(self, vesicles_channel, sample_name="unknown", output_dir=None):
        """Segment vesicles by splitting the full red area into vesicles."""
        if vesicles_channel is None:
            return None

        try:
            print("   Improved vesicle segmentation...")

            try:
                otsu_threshold = filters.threshold_otsu(vesicles_channel)
                binary_otsu = vesicles_channel > otsu_threshold
            except Exception:
                binary_otsu = vesicles_channel > 10

            intensity_mask = vesicles_channel > self.min_vesicle_intensity
            percentile_threshold = np.percentile(vesicles_channel[vesicles_channel > 0], 30) if np.any(vesicles_channel > 0) else 10
            binary_percentile = vesicles_channel > percentile_threshold
            full_red_area = np.logical_and(binary_otsu, intensity_mask)
            full_red_area = np.logical_or(full_red_area, binary_percentile & intensity_mask)

            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            full_red_area = cv2.morphologyEx(full_red_area.astype(np.uint8), cv2.MORPH_CLOSE, kernel_close)
            full_red_area = full_red_area.astype(bool)
            full_red_area = ndimage.binary_fill_holes(full_red_area)

            total_red_pixels = np.sum(full_red_area)
            print(f"      Full red area pixels: {total_red_pixels}")

            if total_red_pixels == 0:
                print("      Warning: no red area for segmentation")
                return np.zeros_like(vesicles_channel, dtype=np.uint8)

            final_vesicles = self.segment_entire_red_area(
                full_red_area,
                vesicles_channel,
                sample_name=sample_name,
                output_dir=output_dir,
            )

            final_labels = measure.label(final_vesicles)
            final_count = np.max(final_labels) if np.max(final_labels) > 0 else 0
            regions = measure.regionprops(final_labels)
            if regions:
                areas = [r.area for r in regions]
                print("   Segmentation result:")
                print(f"      Total vesicles: {final_count}")
                print(f"      Sizes: min={np.min(areas):.1f}, avg={np.mean(areas):.1f}, max={np.max(areas):.1f}")
                print(f"      Coverage: {total_red_pixels}px -> {sum(areas)}px ({sum(areas) / total_red_pixels * 100:.1f}%)")
            else:
                print("   Warning: no vesicles created")

            return final_vesicles

        except Exception as e:
            print(f"   Error in segment_vesicles: {e}")
            return None

    def segment_vesicles_with_criteria(self, vesicles_channel, collagen_channel, sample_name, output_dir=None):
        """Segmentation with criteria and diagnostics-aware propagation to segment_entire_red_area."""
        if vesicles_channel is None:
            return None, []

        try:
            print("      Segmentation with criteria:")
            print(
                f"         Min red-channel intensity (channel 2, 0-255): {self.min_vesicle_intensity}"
            )
            print(f"         Min difference from collagen: {self.intensity_diff_threshold}")

            intensity_mask = vesicles_channel > self.min_vesicle_intensity

            diff_threshold = self.intensity_diff_threshold * 1.5
            if collagen_channel is not None:
                diff_mask = (vesicles_channel.astype(np.float32) - collagen_channel.astype(np.float32)) > diff_threshold
            else:
                diff_mask = np.ones_like(vesicles_channel, dtype=bool)

            combined_mask = intensity_mask & diff_mask

            print("      Mask statistics:")
            print(f"         By intensity (>={self.min_vesicle_intensity}): {np.sum(intensity_mask)} pixels")
            print(f"         By collagen difference (>={diff_threshold:.1f}): {np.sum(diff_mask)} pixels")
            print(f"         Combined: {np.sum(combined_mask)} pixels")

            if np.sum(combined_mask) == 0:
                print("      Warning: no pixels satisfy segmentation criteria")
                return np.zeros_like(vesicles_channel, dtype=np.uint8), []

            if output_dir is None:
                output_dir = getattr(self, "_current_segmentation_output_dir", None)

            vesicles_binary = self.segment_entire_red_area(
                combined_mask,
                vesicles_channel,
                sample_name=sample_name,
                output_dir=output_dir,
            )

            if vesicles_binary is not None and np.sum(vesicles_binary) > 0:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                vesicles_binary = cv2.morphologyEx(vesicles_binary.astype(np.uint8), cv2.MORPH_OPEN, kernel)
                vesicles_binary = (vesicles_binary > 0).astype(np.uint8) * 255

            if vesicles_binary is not None and np.sum(vesicles_binary) > 0:
                labels = measure.label(vesicles_binary)
                regions = measure.regionprops(labels, intensity_image=vesicles_channel)

                vesicle_stats = []
                for i, region in enumerate(regions):
                    mean_intensity = region.mean_intensity
                    if collagen_channel is not None:
                        collagen_in_vesicle = np.mean(collagen_channel[labels == region.label])
                        diff_from_collagen = mean_intensity - collagen_in_vesicle
                    else:
                        collagen_in_vesicle = 0
                        diff_from_collagen = 0

                    vesicle_stats.append({
                        'id': i + 1,
                        'area': region.area,
                        'mean_intensity': mean_intensity,
                        'max_intensity': np.max(vesicles_channel[labels == region.label]),
                        'collagen_in_vesicle': collagen_in_vesicle,
                        'diff_from_collagen': diff_from_collagen,
                        'centroid_y': region.centroid[0],
                        'centroid_x': region.centroid[1],
                    })

                print(f"      Found vesicles: {len(vesicle_stats)}")
                if vesicle_stats:
                    intensities = [v['mean_intensity'] for v in vesicle_stats]
                    print(f"         Intensity range: {min(intensities):.1f} - {max(intensities):.1f}")
                    print(f"         Mean intensity: {np.mean(intensities):.1f}")

                return vesicles_binary, vesicle_stats

            print("      Warning: failed to segment vesicles")
            return np.zeros_like(vesicles_channel, dtype=np.uint8), []

        except Exception as e:
            print(f"      Error in segment_vesicles_with_criteria: {e}")
            return None, []

    def segment_entire_red_area(self, full_red_area, vesicles_channel, sample_name="unknown", output_dir=None):
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
            reasons = {
                'too_small': 0,
                'kept_single': 0,
                'split_disabled': 0,
                'split_success': 0,
                'split_failed': 0,
                'errors': 0,
                'details': [],
            }

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
                    region_intensity = float(region.mean_intensity) if region.mean_intensity is not None else 0.0

                    print(f"      Region {i + 1}: area={region_area}px, circularity={circularity:.3f}")

                    if region_area < self.min_vesicle_size:
                        reasons['too_small'] += 1
                        reasons['details'].append({
                            'id': i + 1,
                            'area': int(region_area),
                            'intensity': region_intensity,
                            'centroid_x': float(region.centroid[1]),
                            'centroid_y': float(region.centroid[0]),
                            'status': 'skipped',
                            'reason': 'too_small',
                        })
                        continue

                    is_cluster = False
                    effective_factor = self.cluster_threshold_factor * 0.8
                    if region_area > self.max_vesicle_size * effective_factor:
                        is_cluster = True
                    if circularity < 0.5 and region_area > self.max_vesicle_size:
                        is_cluster = True


                    if not is_cluster and region_area <= self.max_vesicle_size:
                        final_vesicles_mask[region_mask] = True
                        total_created_vesicles += 1
                        reasons['kept_single'] += 1
                        reasons['details'].append({
                            'id': i + 1,
                            'area': int(region_area),
                            'intensity': region_intensity,
                            'centroid_x': float(region.centroid[1]),
                            'centroid_y': float(region.centroid[0]),
                            'status': 'kept_single',
                            'reason': 'single_region',
                        })
                        print("         Keeping as single vesicle")
                        continue

                    if not self.split_large_clusters:
                        final_vesicles_mask[region_mask] = True
                        total_created_vesicles += 1
                        reasons['split_disabled'] += 1
                        reasons['details'].append({
                            'id': i + 1,
                            'area': int(region_area),
                            'intensity': region_intensity,
                            'centroid_x': float(region.centroid[1]),
                            'centroid_y': float(region.centroid[0]),
                            'status': 'kept_single',
                            'reason': 'split_disabled',
                        })
                        print("         Large cluster splitting disabled, keeping region as-is")
                        continue

                    total_clusters += 1
                    current_mask = region_mask.copy()
                    current_area = int(np.sum(current_mask))
                    best_mask = np.zeros_like(current_mask, dtype=bool)
                    area_ratio = max(current_area / max(self.max_vesicle_size, 1), 1)
                    max_iterations = max(5, int(np.log2(area_ratio)) + 2)
                    max_iterations = min(max_iterations, self.max_split_iterations * 2)
                    iteration = 0

                    print(f"         Cluster detected: area={current_area}px, circularity={circularity:.3f}")
                    print(f"         Adaptive max iterations: {max_iterations}")

                    while iteration < max_iterations and current_area > self.max_vesicle_size * self.split_iterations_factor:
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
                        large_regions = [area for area in areas_after if area > self.max_vesicle_size * self.split_iterations_factor]

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

                    if cluster_count > 0:
                        reasons['split_success'] += 1
                        reasons['details'].append({
                            'id': i + 1,
                            'area': int(region_area),
                            'intensity': region_intensity,
                            'centroid_x': float(region.centroid[1]),
                            'centroid_y': float(region.centroid[0]),
                            'status': 'split_success',
                            'reason': f'created_{cluster_count}_vesicles',
                        })
                    else:
                        reasons['split_failed'] += 1
                        reasons['details'].append({
                            'id': i + 1,
                            'area': int(region_area),
                            'intensity': region_intensity,
                            'centroid_x': float(region.centroid[1]),
                            'centroid_y': float(region.centroid[0]),
                            'status': 'split_failed',
                            'reason': 'no_vesicles_created',
                        })

                    print(f"         Total split iterations: {iteration}")
                    print(f"         Created vesicles from region: {cluster_count}")

                except Exception as region_error:
                    reasons['errors'] += 1
                    reasons['details'].append({
                        'id': i + 1,
                        'area': int(region.area) if hasattr(region, 'area') else 0,
                        'intensity': float(region.mean_intensity) if hasattr(region, 'mean_intensity') and region.mean_intensity is not None else 0.0,
                        'centroid_x': float(region.centroid[1]) if hasattr(region, 'centroid') else 0.0,
                        'centroid_y': float(region.centroid[0]) if hasattr(region, 'centroid') else 0.0,
                        'status': 'error',
                        'reason': str(region_error),
                    })
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

                if output_dir is not None and sample_name != "unknown":
                    self.write_segmentation_diagnostics(
                        sample_name,
                        reasons,
                        len(red_regions),
                        total_created_vesicles,
                        output_dir,
                    )

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

            cluster_labels = measure.label(cluster_mask)
            cluster_regions = measure.regionprops(cluster_labels)
            if cluster_regions:
                cluster_region = max(cluster_regions, key=lambda region: region.area)
                perimeter = cluster_region.perimeter
                circularity = 4 * np.pi * cluster_area / (perimeter ** 2) if perimeter > 0 else 0.0
            else:
                circularity = 0.0

            print(f"            Source cluster size: {cluster_area}px")
            print(f"            Bounding box: {width}x{height}")
            print(f"            Target vesicle count: {optimal_count}")
            print(f"            Cluster circularity: {circularity:.3f}")

            if circularity < 0.3 and cluster_area > self.max_vesicle_size * 1.5:
                print("            Low circularity cluster detected, prioritizing uniform grid")
                methods = [
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
                            vesicles_channel,
                        ),
                    ),
                ]
            else:
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
                            vesicles_channel,
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
