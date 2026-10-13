import unittest
from delayed_audio import AudioFrame, DelayedAudio, FrameParser


def adts(payload=b'1234567890'):
    size = 7+len(payload)
    return bytes((255,241,76,128|(size>>11), (size>>3)&255, ((size&7)<<5)|31,252))+payload


class DelayedAudioTests(unittest.TestCase):
    def test_frame_packetization_and_exact_clock(self):
        audio = adts()*100
        parser = FrameParser()
        frames = []
        for i in range(0,len(audio),13):
            frames += parser.feed(audio[i:i+13])
        self.assertEqual(b''.join(f.data for f in frames), audio)
        self.assertAlmostEqual(frames[-1].end, 100*1024/48000)
        self.assertEqual(len(frames),100)

    def test_id3_and_junk_are_not_timed_as_audio(self):
        parser = FrameParser()
        frames = parser.feed(b'ID3\x04\x00\x00\x00\x00\x00\x04junk'+adts()*2)
        self.assertEqual(len(frames),2)
        self.assertEqual(frames[0].start,0)

    def test_delay_outage_drain_and_reconnect(self):
        buffer = DelayedAudio(30)
        buffer.new_segment()
        frames = [AudioFrame(i,i+1,str(i).encode()) for i in range(40)]
        buffer.append(frames,40)
        self.assertEqual([f.start for f in buffer.release(40)], list(range(10)))
        self.assertEqual(buffer.bootstrap(), [frames[9]])
        self.assertEqual([f.start for f in buffer.release(43)], [10,11,12])
        self.assertEqual(len(buffer.release(70)),27)
        buffer.new_segment()
        buffer.append([AudioFrame(40,41,b'new')], 100)
        self.assertEqual(buffer.release(129),[])
        self.assertEqual(buffer.release(130)[0].data,b'new')

    def test_startup_burst_uses_media_delay_without_extra_wait(self):
        buffer = DelayedAudio(30)
        buffer.append([AudioFrame(i,i+1,b'x') for i in range(35)],100)
        self.assertEqual(len(buffer.release(100)),5)
        self.assertEqual(len(buffer.release(101)),1)
        for i in range(35,300):
            buffer.append([AudioFrame(i,i+1,b'x')],100+i-34)
            buffer.release(100+i-34)
        self.assertLessEqual(len(buffer.pending),30)
        self.assertLessEqual(len(buffer.released),11)

