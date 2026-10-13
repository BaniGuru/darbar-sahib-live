import asyncio
import unittest
from unittest.mock import patch

import server
from delayed_audio import AudioFrame, DelayedAudio


class PlaybackTests(unittest.TestCase):
    def setUp(self):
        self.sid = '11111111-1111-4111-8111-111111111111'
        buffer = DelayedAudio(10)
        buffer.append([AudioFrame(i, i+1, b'audio') for i in range(40)],40)
        buffer.release(40)
        for name,value in (('_delayed_audio',buffer), ('_stream_wall_started_at',1000),
                           ('_audio_sessions',{}), ('_audio_subscribers',set()),
                           ('_pankti_timeline',[]), ('_debug_timed_final_words',[]),
                           ('_debug_timed_partial_words',[]),
                           ('PLAYBACK_DELAY_SECONDS',10)):
            p=patch.object(server,name,value);p.start();self.addCleanup(p.stop)

    def test_listener_never_receives_future_match_or_transcript(self):
        old = dict(punjabi='heard',english='',legacy_gurmukhi='',shabad_id='1',pankti_id='1')
        future = {**old, 'punjabi':'future', 'pankti_id':'2'}
        server._pankti_timeline[:] = [(1010,old),(1040,future)]
        server._debug_timed_final_words[:] = [dict(text='heard',start=10,end=11),dict(text='future',start=40,end=41)]
        server._audio_sessions[self.sid] = dict(last_seen=0,first_audio_at=1029,generation=1)
        state = server.live_state(self.sid,0)
        self.assertEqual(state['pankti'],'heard')
        self.assertEqual(state['playback_text'],'heard')
        later = server.live_state(self.sid,12)
        self.assertEqual(later['pankti'],'future')
        self.assertEqual(later['playback_text'],'heard future')
        self.assertNotEqual(state['revision'],later['revision'])
        self.assertEqual(server.live_state(self.sid,12,known_revision=later['revision']).status_code,204)

    def test_reconnect_resets_anchor_and_closes_previous_queue(self):
        async def check():
            first = await server.live_audio(self.sid)
            await anext(first.body_iterator)
            session = server._audio_sessions[self.sid]
            session['first_audio_at'] = 900
            previous_queue = session['queue']
            second = await server.live_audio(self.sid)
            self.assertIsNone(session['first_audio_at'])
            self.assertIsNone(previous_queue.get_nowait())
            await anext(second.body_iterator)
            self.assertEqual(session['first_audio_at'],1029)
            self.assertEqual(session['generation'],2)
            await first.body_iterator.aclose()
            await second.body_iterator.aclose()
        asyncio.run(check())

    def test_confirmed_match_backdates_to_first_supported_word(self):
        message = dict(committed_words=[dict(text='ਨਾਮ',start=2,end=3),dict(text='ਜਪੋ',start=5,end=6)])
        self.assertEqual(server._matched_word_offset(message,'ਨਾਮ ਜਪੋ',2),6)
        self.assertEqual(server._matched_word_offset(message,'ਨਾਮ ਜਪੋ',2,'ਨਾਮ ਜਪੋ'),2)

    def test_listener_count_tracks_stream_connections_and_invalidates_revision(self):
        async def check():
            initial = server.live_state(self.sid)
            self.assertEqual(initial['listener_count'], 0)
            first = await server.live_audio(self.sid)
            await anext(first.body_iterator)
            listening = server.live_state(self.sid)
            self.assertEqual(listening['listener_count'], 1)
            self.assertNotEqual(initial['revision'], listening['revision'])
            second = await server.live_audio(self.sid)
            await anext(second.body_iterator)
            self.assertEqual(server.live_state(self.sid)['listener_count'], 1)
            await first.body_iterator.aclose()
            await second.body_iterator.aclose()
            stopped = server.live_state(self.sid)
            self.assertEqual(stopped['listener_count'], 0)
            self.assertNotEqual(listening['revision'], stopped['revision'])
        asyncio.run(check())

