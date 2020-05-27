import os
import sys

sdk_path = os.path.join(os.path.dirname(__file__), 'sdk_face_samples_Python3')
sys.path.append(sdk_path)
os.environ['LD_LIBRARY_PATH'] = os.path.join(sdk_path, 'core/lib:$LD_LIBRARY_PATH')
# from core.internal_sdk import *
from core.load_license import *
from utils.utils_package import *


def my_licence_check():
    if sdk_protector_has_license(b"Sensetime") != 0:
        license_file = os.path.join(sdk_path, "core/SENSETIME_1F2E5FAA-DBF7-41FA-AC2A-D6D8A52F18CD.lic")
        with open(license_file, "r") as f:
            license = f.read()
        sdk_protector_add_license("Sensetime", license, None)


class LandmarkDetector:
    def __init__(self):
        self.Track_Strategy = "SYNC_TRACK"
        self.detect_interval = 20
        self.max_face_count = 10  # 1~32
        self.align_threshold = 1000  # 0~10000

        ## set log level, easy debug
        cv_common_set_log_level(c_int(1))

        ## add license
        my_licence_check()
        self.m_detect_hunter = os.path.join(sdk_path, "models/detection/M_Detect_Hunter_SmallFace_Gray_9.2.3.model")
        self.m_track_106 = os.path.join(sdk_path,
                                        "models/landmark_tracking/Occlusion/M_SenseAR_LandmarkTrackocclusion_Face106pt_FastV1_1.3.0.model")
        ## load model and create_handle
        self.model_detect = cv_model_t()
        st_result = cv_common_load_model(self.m_detect_hunter, pointer(self.model_detect))
        assert st_result == CV_OK, "fail to load hunter model: {}".format(st_result)

        self.handle_detect = cv_handle_t()
        st_result = cv_common_detection_hunter_create(self.model_detect, pointer(self.handle_detect))
        assert st_result == CV_OK, "fail to init detect_hunter handle: {}".format(st_result)
        cv_common_unload_model(self.model_detect)

        model_align = cv_model_t()
        st_result = cv_common_load_model(self.m_track_106, pointer(model_align))
        assert st_result == CV_OK, "fail to load align_106 model: {}".format(st_result)

        self.handle_track = cv_handle_t()
        config = c_uint(0)
        if self.Track_Strategy == "SYNC_TRACK":
            st_result = cv_common_tracking_compactocclusion_create(model_align, self.handle_detect, config,
                                                                   pointer(self.handle_track))
            assert st_result == CV_OK, "fail to init track handle: {}".format(st_result)
        elif self.Track_Strategy == "ASYNC_TRACK_DEADLINE":
            st_result = cv_common_tracking_compactocclusion_create(model_align, self.handle_detect,
                                                                   CV_COMMON_TRACKING_ASYNC | CV_COMMON_TRACKING_ASYNC_DETECTDEADLINE,
                                                                   pointer(self.handle_track))
            assert st_result == CV_OK, "fail to init track handle: {}".format(st_result)
            new_val = c_int()
            st_result = cv_common_tracking_compact_config(self.handle_track,
                                                          CV_COMMON_TRACKING_CONF_ASYNC_DETECTDEADLINE_DURATION, 2000,
                                                          pointer(new_val))
            assert st_result == CV_OK, "fail to init track deadline config: {}".format(st_result)
        else:
            raise NotImplementedError()
        cv_common_unload_model(model_align)

        print("success to init all handle")
        new_val = c_int()
        st_result = cv_common_tracking_compact_config(self.handle_track, CV_COMMON_TRACKING_CONF_DETECTINTERVAL,
                                                      self.detect_interval, pointer(new_val))
        assert st_result == CV_OK, "fail to init track config 1: {}".format(st_result)
        st_result = cv_common_tracking_compact_config(self.handle_track, CV_COMMON_TRACKING_CONF_LIMIT,
                                                      self.max_face_count, pointer(new_val))
        assert st_result == CV_OK, "fail to init track config 2: {}".format(st_result)
        st_result = cv_common_tracking_compact_trackbase_config(self.handle_track, CV_COMMON_TRACKING_CONF_THRESHOLD,
                                                                self.align_threshold, pointer(new_val))
        assert st_result == CV_OK, "fail to init track config 3: {}".format(st_result)

        print("success to init all track config")

    def detect_landmarks(self, color_img):
        frame = color_img
        current_time = time.time()
        tv_sec = int(math.modf(current_time)[1])
        tv_usec = int(('%.8f' % round(math.modf(current_time)[0], 8)).split(".")[-1])
        # print(current_time,tv_sec,tv_usec)
        cv_image = cv_image_t()
        cv_image.height = frame.shape[0]
        cv_image.width = frame.shape[1]
        cv_image.data = frame.ctypes.data_as(POINTER(c_uint8))
        cv_image.stride = cv_image.width * 3  ## how many bytes each row. jpg has 3 channels.
        cv_image.pixel_format = CV_PIX_FMT_BGR888
        cv_image.time_stamp.tv_sec = int(tv_sec)
        cv_image.time_stamp.tv_usec = int(tv_usec)
        # img_out = frame
        p_face = POINTER(cv_target_t)()
        face_count = c_int(0)
        st_result = cv_common_tracking_compact_track(self.handle_track, pointer(cv_image), CV_ORIENTATION_0,
                                                     pointer(p_face), pointer(face_count))
        assert st_result == CV_OK, "cv_common_tracking_compact_track error : {}".format(st_result)
        p_face = ctypes.cast(p_face, POINTER(cv_target_t * face_count.value))
        if face_count.value == 0:
            return dict(lmk_success=False)
        elif face_count.value > 1:
            print('Warning: more than one face, use the first face')
        align_result = p_face.contents[0].landmarks
        # detect_result = p_face.contents[0].detect_result
        score = align_result.score
        landmarks = align_result.points_array
        points_count = align_result.points_count
        occlusion = align_result.occlusion  # 0 indicate if landmark is occluded
        # rect = detect_result.rect

        occlusion_np = np.array([occlusion[j] for j in range(points_count)])
        landmarks_np = np.array([[landmarks[j].x, landmarks[j].y] for j in range(points_count)])
        res = dict(occlusion=occlusion_np, landmarks=landmarks_np, lmk_count=points_count, lmk_score=score,
                   lmk_success=True)
        return res
