import sys
import torch


def clothes_mapping(predictions):
    mapping_rules = {
                        0: 3,
                        1: 1,
                        2: 3,
                        3: 3,
                        4: 2,
                        5: 0,
                        6: 3,
                        7: 0,
                        8: 1,
                        9: 4,
                    }
    max_val = 4
    out_predictions = torch.zeros([predictions.size(0), max_val + 1])
    for k, prediction in enumerate(predictions):
        out = [None for _ in range(max_val + 1)]
        for i, val in enumerate(prediction):
            key = mapping_rules[i]
            if out[key] is None:
                out[key] = val
            else:
                out[key] = max(out[key], val)
        out_predictions[k] = torch.tensor(out)

    return out_predictions.cuda()


def mapping(predictions: list, mapping_domains: list, attributes: list):
    for i, attribute in enumerate(attributes):
        if attribute in mapping_domains:
            if attribute == 'Type':
                predictions[i] = clothes_mapping(predictions[i])
            else:
                sys.exit(f"No mapping for {attribute}")
    return predictions
