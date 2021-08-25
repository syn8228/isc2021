import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from isc.metrics import evaluate, Metrics, PredictedMatch, print_metrics
from isc.io import read_ground_truth, read_predictions, write_predictions, read_descriptors
from isc.descriptor_matching import match_and_make_predictions, knn_match_and_make_predictions
from typing import Optional

import h5py

import faiss


def generate_match_list(db_description, query_description, threshold=3.0, pth=None):
    match_list = []
    db_image_ids, vectors_db = read_descriptors(db_description)
    query_image_ids, vectors_query = read_descriptors(query_description)
    for i in range(len(vectors_query)):
        vec_q = vectors_query[i]
        differences = vectors_db - vec_q
        l2_distances = np.linalg.norm(differences, axis=1)
        matched_index = np.argsort(l2_distances)[:10]
        for inx in matched_index:
            if l2_distances[inx] > threshold:
                pass
            else:
                score = 1/(l2_distances[inx] + np.finfo(float).eps)
                match_list.append((query_image_ids[i], db_image_ids[inx], score))
    name = ['query_id', 'reference_id', 'score']
    matched = pd.DataFrame(columns=name, data=match_list)
    matched.to_csv(pth, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        group.add_argument(*args, **kwargs)

    group = parser.add_argument_group("input")
    aa("--db_descs", nargs='*', help="database descriptor file in HDF5 format")
    aa("--query_descs", nargs='*', help="query descriptor file in HDF5 format")
    aa('--threshold', default=3.0, type=float, help="learning rate")

    group = parser.add_argument_group("output")
    aa("--predict_filepath", default="", help="output file for prediction")

    args = parser.parse_args()

    generate_match_list(args.db_descs, args.query_descs, threshold=args.threshold, pth=args.predict_filepath)


