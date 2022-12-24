import numpy as np
import cv2 as cv

def generate_priors( input_w, input_h):
    # Calculate shapes of different scales according to the shape of input image
    feature_map_2nd = (int((input_w + 1) / 2) // 2, int((input_h + 1) / 2) // 2)
    feature_map_3rd = (int(feature_map_2nd[0] / 2), int(feature_map_2nd[1] / 2))
    feature_map_4th = (int(feature_map_3rd[0] / 2), int(feature_map_3rd[1] / 2))
    feature_map_5th = (int(feature_map_4th[0] / 2), int(feature_map_4th[1] / 2))
    feature_map_6th = (int(feature_map_5th[0] / 2), int(feature_map_5th[1] / 2))

    feature_map_sizes = [feature_map_3rd, feature_map_4th, feature_map_5th, feature_map_6th]

    # Fixed params for generating priors
    min_sizes = [[10.0,  16.0,  24.0],
                 [32.0,  48.0],
                 [64.0,  96.0],
                 [128.0, 192.0, 256.0]]
    steps = [8, 16, 32, 64]

    # Generate priors
    priors = []
    for i in range(len(feature_map_sizes)):
        feature_map_size = feature_map_sizes[i]
        min_size = min_sizes[i]

        for _h in range(feature_map_size[1]):
            for _w in range(feature_map_size[0]):
                for j in range(len(min_size)):
                    s_kx = min_size[j] / input_w
                    s_ky = min_size[j] / input_h

                    cx = (_w + 0.5) * steps[i] / input_w
                    cy = (_h + 0.5) * steps[i] / input_h

                    prior = (cx, cy, s_kx, s_ky)
                    priors.append(prior)
    return priors


def post_process(output_blobs, input_w, input_h):

    # Extract from output_blobs
    loc = output_blobs[0]
    conf = output_blobs[1]
    iou = output_blobs[2]

    # Decode from deltas and priors
    variance = [0.1, 0.2]
    loc_v = loc.flatten()
    conf_v = conf.flatten()
    iou_v = iou.flatten()
    faces = []
    # (tl_x, tl_y, w, h, re_x, re_y, le_x, le_y, nt_x, nt_y, rcm_x, rcm_y, lcm_x, lcm_y, score)
    # 'tl': top left point of the bounding box
    # 're': right eye, 'le': left eye
    # 'nt':  nose tip
    # 'rcm': right corner of mouth, 'lcm': left corner of mouth
    face = np.empty((1, 15))
    priors = generate_priors( input_w, input_h)
    for i in range(len(priors)):
        # Get score
        cls_score = conf_v[i*2+1]
        iou_score = iou_v[i]
        # Clamp
        if iou_score < 0:
            iou_score = 0
        elif iou_score > 1:
            iou_score = 1
        score = np.sqrt(cls_score * iou_score)
        face[0, 14] = score

        # Get bounding box
        cx = (priors[i][0] + loc_v[i*14+0] * variance[0] * priors[i][2]) * input_w
        cy = (priors[i][1] + loc_v[i*14+1] * variance[0] * priors[i][3]) * input_h
        w  = priors[i][2] * np.exp(loc_v[i*14+2] * variance[0]) * input_w
        h  = priors[i][3] * np.exp(loc_v[i*14+3] * variance[1]) * input_h
        x1 = cx - w / 2
        y1 = cy - h / 2
        face[0, 0] = x1
        face[0, 1] = y1
        face[0, 2] = w
        face[0, 3] = h

        # Get landmarks
        face[0, 4] = (priors[i][0] + loc_v[i*14+4] * variance[0] * priors[i][2]) * input_w  # right eye, x
        face[0, 5] = (priors[i][1] + loc_v[i*14+5] * variance[0] * priors[i][3]) * input_h  # right eye, y
        face[0, 6] = (priors[i][0] + loc_v[i*14+6] * variance[0] * priors[i][2]) * input_w  # left eye, x
        face[0, 7] = (priors[i][1] + loc_v[i*14+7] * variance[0] * priors[i][3]) * input_h  # left eye, y
        face[0, 8] = (priors[i][0] + loc_v[i*14+8] * variance[0] * priors[i][2]) * input_w  # nose tip, x
        face[0, 9] = (priors[i][1] + loc_v[i*14+9] * variance[0] * priors[i][3]) * input_h  # nose tip, y
        face[0, 10] = (priors[i][0] + loc_v[i*14+10] * variance[0] * priors[i][2]) * input_w  # right corner of mouth, x
        face[0, 11] = (priors[i][1] + loc_v[i*14+11] * variance[0] * priors[i][3]) * input_h  # right corner of mouth, y
        face[0, 12] = (priors[i][0] + loc_v[i*14+12] * variance[0] * priors[i][2]) * input_w  # left corner of mouth, x
        face[0, 13] = (priors[i][1] + loc_v[i*14+13] * variance[0] * priors[i][3]) * input_h  # left corner of mouth, y

        faces.append(face)

    if len(faces) > 1:
        # Retrieve boxes and scores
        face_boxes = []
        face_scores = []
        for r_idx in range(len(faces)):
            face_boxes.append((int(faces[r_idx][0, 0]),
                               int(faces[r_idx][0, 1]),
                               int(faces[r_idx][0, 2]),
                               int(faces[r_idx][0, 3])))
            face_scores.append(faces[r_idx][0, 14])

        keep_idx = []
        # dnn.NMSBoxes(face_boxes, face_scores, score_threshold, nms_threshold, keep_idx, 1.0, top_k)
        keep_idx = cv.dnn.NMSBoxes(face_boxes, face_scores, 0.5, 0.3, top_k=5000)
        # Get NMS results
        nms_faces = []
        print(len(keep_idx))
        print(priors[0])
        
        return 1

    #     for idx in keep_idx:
    #         nms_faces.append(faces[idx])
    #     return np.vstack(np.array(nms_faces))
    # else:
    #     return np.vstack(np.array(faces))

if __name__ == "__main__":

    conf = np.loadtxt("./data/conf.out")
    loc = np.loadtxt("./data/loc.out")
    iou = np.loadtxt("./data/iou.out")

    output_blobs = [loc, conf, iou]

    result = post_process(output_blobs, 160, 120)