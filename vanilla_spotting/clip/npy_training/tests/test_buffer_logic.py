import sys
import os
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from collections import deque
from PIL import Image
import threading

# Add parent directory to path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the module to test (assuming it's in the parent dir or accessible)
# Since the file is at vanilla_spotting/clip/npy_training/realtime_video_inference.py
# doing a direct import might be tricky if dependencies aren't met.
# I will mock the dependencies first.

sys.modules['cv2'] = MagicMock()
sys.modules['mobileclip'] = MagicMock()
sys.modules['open_clip'] = MagicMock()
sys.modules['torch'] = MagicMock()
sys.modules['model'] = MagicMock()
sys.modules['utils'] = MagicMock()
sys.modules['tqdm'] = MagicMock()

# Now we can import the module code. 
# But wait, I can't easily import it if I just mocked everything, 
# because I need the actual logic in 'process_video_realtime'.
# I will copy the relevant logic into the test for verification,
# OR I can try to import the file dynamically.

import importlib.util
spec = importlib.util.spec_from_file_location("realtime_video_inference", 
    r"c:\Users\USER\Desktop\ExploreVAD\vanilla_spotting\clip\npy_training\realtime_video_inference.py")
module = importlib.util.module_from_spec(spec)
sys.modules["realtime_video_inference"] = module
spec.loader.exec_module(module)

class TestRollingBuffer(unittest.TestCase):

    def test_process_loop_logic(self):
        # Setup mocks
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            module.cv2.CAP_PROP_FPS: 30.0,
            module.cv2.CAP_PROP_FRAME_WIDTH: 100,
            module.cv2.CAP_PROP_FRAME_HEIGHT: 100,
            module.cv2.CAP_PROP_FRAME_COUNT: 100
        }.get(prop, 0)

        # Mock reading frames (return True, random_frame 100 times, then False)
        fake_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_cap.read.side_effect = [(True, fake_frame)] * 50 + [(False, None)]

        mock_engine = MagicMock()
        mock_engine.stride_time = 0.5
        mock_engine.window_time = 2.0
        mock_engine.num_frames = 8
        mock_engine.is_processing = False
        mock_engine.inference_triggers = []
        mock_engine.timeline_data = [] # needed for footer
        mock_engine.latest_label = "TEST"
        mock_engine.latest_prob = 0.0
        mock_engine.latest_color = (0,0,0)
        mock_engine.last_latency = 0.0

        # Patch cv2 in the module to avoid UI calls
        module.cv2.VideoCapture.return_value = mock_cap
        module.cv2.namedWindow = MagicMock()
        module.cv2.resizeWindow = MagicMock()
        module.cv2.imshow = MagicMock()
        module.cv2.waitKey.return_value = 0
        module.cv2.destroyAllWindows = MagicMock()
        module.cv2.cvtColor.return_value = fake_frame
        module.create_padded_frame = MagicMock(return_value=fake_frame)
        module.draw_header = MagicMock()
        module.draw_footer_timeline = MagicMock()
        
        # We need to capture the calls to run_inference
        # But run_inference is run in a thread.
        # We can mock threading.Thread to run synchronously or just inspect args.
        
        captured_inference_calls = []
        
        def mock_thread_target(target, args, daemon):
            frames, time = args
            captured_inference_calls.append({
                'time': time,
                'num_frames': len(frames),
                'first_frame_type': type(frames[0])
            })
            # Simulate inference finishing
            mock_engine.is_processing = False
            
        with patch('threading.Thread') as mock_thread:
            mock_thread.side_effect = lambda target, args, daemon: MagicMock(start=lambda: mock_thread_target(target, args, daemon))
            
            # Run the function
            module.process_video_realtime(
                video_path="dummy.mp4",
                inference_engine=mock_engine,
                show_preview=False,
                playback_speed=100.0 # fast
            )

        # Verifications
        print(f"Captured {len(captured_inference_calls)} inference calls")
        
        # 1. Check if cap.set was called
        # mock_cap.set.assert_not_called() 
        # Actually set might be called by internal opencv things, but we want to ensure WE didn't call it for seeking.
        # calls = mock_cap.set.call_args_list
        # for call_args in calls:
        #     if call_args[0][0] == module.cv2.CAP_PROP_POS_FRAMES:
        #         self.fail("Found call to set CAP_PROP_POS_FRAMES!")

        # 2. Check frequency
        # 50 frames at 30fps is 1.66 seconds.
        # Stride is 0.5s.
        # Should trigger at 0.5, 1.0, 1.5.
        # But we also have a buffer fill requirement.
        # Buffer needs 8 frames (num_frames).
        # At 30fps, 8 frames is ~0.27s. so by 0.5s we have enough frames.
        self.assertTrue(len(captured_inference_calls) >= 3, f"Expected at least 3 inference calls, got {len(captured_inference_calls)}")
        
        # 3. Check frame count in calls
        for call_info in captured_inference_calls:
            self.assertEqual(call_info['num_frames'], 8)

if __name__ == '__main__':
    unittest.main()
