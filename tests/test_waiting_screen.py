import asyncio
import json
import time
import unittest
from unittest.mock import patch

import server
from bani_matching import OrderedBaniMatcher


class WaitingScreenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        server._build_pankti_index()
        cls.rows={r['pankti_id']:r for r in server._pankti_rows}
        cls.routes=server._bani_matcher.routes

    def setUp(self):
        values=dict(_bani_matcher=None,_active_shabad_id='NJM',
                    _live_match=server._active_match_payload(self.rows['N43J'],1,2),_live_match_at=time.time(),
                    _pending_shabad_id=None,_pending_shabad_hits=0,_pending_shabad_update=None,
                    _pending_pankti_id=None,_pending_pankti_hits=0,_pending_pankti_audio_time=None,
                    _debug_final_text='',_debug_partial_text='',_debug_timed_final_words=[],_debug_timed_partial_words=[],
                    _debug_recompute_from=None,_pankti_timeline=[],_playback_pankti_timeline=[],
                    _stream_wall_started_at=1000,_greeting_audio_cutoff=None,_last_greeting_key=None,
                    _waiting_context=None,
                    _audio_sessions={},_audio_subscribers=set())
        p=patch.multiple(server,**values);p.start();self.addCleanup(p.stop)
        self.sid='11111111-1111-4111-8111-111111111111'
        server._audio_sessions[self.sid]=dict(last_seen=time.monotonic(),first_audio_at=1000,generation=1)

    def send(self,text,start=10,offset=0,audio_active=True):
        words=[dict(text=w,start=start+i,end=start+i+.8) for i,w in enumerate(text.split())]
        message=dict(type='partial',partial_text=text,partial_words=words,committed_words=[],window_end=words[-1]['end'],audio_active=audio_active)
        class Socket:
            async def __aiter__(self):yield json.dumps(message,ensure_ascii=False)
        with patch.object(server.logger,'info'):
            asyncio.run(server._handle_asr_messages(Socket(),offset))

    def test_timeout_is_60_seconds_from_latest_supported_word(self):
        old={k:v for k,v in server._live_match.items() if not k.startswith('_')}
        server._pankti_timeline[:]=[(1000,{**old,'matched_until':1005})]
        server._prepare_pankti_timeline()
        self.assertFalse(server.live_state(self.sid,64.999)['waiting_for_shabad'])
        self.assertEqual(server.live_state(self.sid,65)['waiting_reason'],'no-shabad')

    def test_never_matched_stream_times_out_and_exposes_future_boundary(self):
        server._prepare_pankti_timeline()
        self.assertFalse(server.live_state(self.sid,59.999)['waiting_for_shabad'])
        early=server.live_state(self.sid,55)
        self.assertEqual(early['pankti_schedule'][0]['audio_timestamp'],1060)
        self.assertEqual(server.live_state(self.sid,60)['waiting_reason'],'no-shabad')

    def test_idle_asr_cannot_keep_an_old_pankti_alive(self):
        old=dict(server._live_match)
        server._pankti_timeline[:]=[(1000,{**old,'matched_until':1005})]
        self.send(self.rows['N43J']['speech'],start=20,audio_active=False)
        self.assertFalse(server.live_state(self.sid,64.99)['waiting_for_shabad'])
        self.assertEqual(server.live_state(self.sid,65)['waiting_reason'],'no-shabad')

    def test_simran_uses_waiting_and_never_the_four_waheguru_shabad(self):
        server._pankti_timeline[:]=[(1000,dict(server._live_match))]
        self.send('ਵਾਹਿਗੁਰੂ ਵਾਹਿਗੁਰੂ ਵਾਹਿਗੁਰੂ ਵਾਹਿ ਜੀਉ')
        self.assertFalse(server.live_state(self.sid,9.9)['waiting_for_shabad'])
        self.assertEqual(server.live_state(self.sid,10)['waiting_reason'],'simran')
        self.assertIsNone(server._live_match)
        self.assertEqual(len([line for _,line in server._pankti_timeline if line.get('waiting_for_shabad')]),1)

    def test_ardas_is_loaded_from_bani_membership_and_waits_at_first_word(self):
        self.assertIn('PSRF',server._ardas_pankti_ids)
        self.assertGreaterEqual(len(server._ardas_pankti_ids),27)
        self.send('ਪ੍ਰਿਥਮ ਭਗੌਤੀ ਸਿਮਰਿ ਕੈ ਗੁਰ ਨਾਨਕ ਲਈਂ ਧਿਆਇ')
        self.assertEqual(server.live_state(self.sid,10)['waiting_reason'],'ardas')
        self.assertIsNone(server._live_match)

    def test_long_ardas_lines_do_not_leave_waiting_through_common_words(self):
        for index,text in enumerate(['ਹੇ ਅਕਾਲ ਪੁਰਖ ਆਪਣੇ ਪੰਥ ਦੇ ਸਦਾ ਸਹਾਈ ਦਾਤਾਰ ਜੀਓ',
                                     'ਨਾਨਕ ਨਾਮ ਚੜ੍ਹਦੀ ਕਲਾ ਤੇਰੇ ਭਾਣੇ ਸਰਬੱਤ ਦਾ ਭਲਾ']):
            self.send(text,start=10+index*20)
            self.assertIsNone(server._live_match)

    def test_recorded_ardas_names_do_not_resume_unrelated_shabads(self):
        self.send('ਪ੍ਰਿਥਮ ਭਗੌਤੀ ਸਿਮਰਿ ਕੈ ਗੁਰ ਨਾਨਕ ਲਈਂ ਧਿਆਇ')
        # Actual ASR near 03:42 in the October 14 recording. Repeated ਹਰਿ
        # previously counted as four words and selected unrelated DX8N.
        self.send('ਰਾਮਦਾਸੈ ਹੋਇ ਸਹਾਈ ਹਰਿ ਜਨ ਹਰਿ ਗੋਵਿੰਦਾ ਨਾਉ ਸਮਰਾਉ ਸ੍ਰੀ ਹਰਿ ਰਾਇਓ',start=30)
        self.assertIsNone(server._live_match)
        self.assertTrue(server._waiting_context)
        self.send('ਕਾਟੇ ਸੁ ਪਾਪ ਨਰਹੁ ਕੇ ਗੁਰ ਰਾਮਦਾਸ ਜਿਨ ਪਾਇਓ',start=50)
        self.assertEqual(server._live_match['pankti_id'],'BKS1')
        self.assertFalse(server._waiting_context)

    def test_ardas_can_resume_a_bani_without_an_end_greeting(self):
        server._bani_matcher=OrderedBaniMatcher(self.routes,server._token_similarity,server._normalise_query)
        self.send('ਪ੍ਰਿਥਮ ਭਗੌਤੀ ਸਿਮਰਿ ਕੈ ਗੁਰ ਨਾਨਕ ਲਈਂ ਧਿਆਇ')
        self.send('ਥਾਲ ਵਿਚਿ ਤਿੰਨਿ ਵਸਤੂ ਪਈਓ ਸਤੁ ਸੰਤੋਖੁ ਵੀਚਾਰੋ',start=40)
        # The internal bani cursor can already predict its next line; the
        # listener must receive the heard line at its first word.
        self.assertEqual(server.live_state(self.sid,40)['pankti_id'],'WB4V')
        self.assertFalse(server._waiting_context)

    def test_fateh_tail_after_simran_keeps_waiting(self):
        self.send('ਵਾਹਿਗੁਰੂ ਵਾਹਿਗੁਰੂ ਜੀ ਕੀ ਫਤਹਿ')
        self.assertIsNone(server._live_match)
        self.assertTrue(server.live_state(self.sid,13)['waiting_for_shabad'])

    def test_full_end_greeting_releases_ardas_protection(self):
        self.send('ਪ੍ਰਿਥਮ ਭਗੌਤੀ ਸਿਮਰਿ ਕੈ ਗੁਰ ਨਾਨਕ ਲਈਂ ਧਿਆਇ')
        self.assertTrue(server._waiting_context)
        self.send('ਵਾਹਿਗੁਰੂ ਜੀ ਕਾ ਖਾਲਸਾ ਵਾਹਿਗੁਰੂ ਜੀ ਕੀ ਫਤਹਿ',start=40)
        self.assertFalse(server._waiting_context)

    def test_simran_does_not_consume_a_later_shabad_in_same_batch(self):
        self.send('ਵਾਹਿਗੁਰੂ ਵਾਹਿਗੁਰੂ ਕਾਟੇ ਸੁ ਪਾਪ ਨਰਹੁ ਕੇ ਗੁਰ ਰਾਮਦਾਸ ਜਿਨ ਪਾਇਓ')
        self.assertEqual(server._live_match['pankti_id'],'BKS1')
        self.assertTrue(server.live_state(self.sid,11)['waiting_for_shabad'])
        self.assertFalse(server.live_state(self.sid,20)['waiting_for_shabad'])

    def test_special_waiting_uses_original_layout_without_a_raagi(self):
        self.send('ਵਾਹਿਗੁਰੂ ਵਾਹਿਗੁਰੂ')
        for position in (20, 70, 900):
            state=server.live_state(self.sid,position)
            self.assertTrue(state['waiting_for_shabad'])
            self.assertNotIn('raagi',state)
            self.assertNotIn('raagi_artwork_url',state)

    def test_closing_pauris_wait_and_do_not_show_following_lines(self):
        for text in ['ਵਡੀ ਹੂ ਵਡਾ ਅਪਾਰੁ ਤੇਰਾ ਮਰਤਬਾ',
                     'ਪਵਣੁ ਗੁਰੂ ਪਾਣੀ ਪਿਤਾ ਮਾਤਾ ਧਰਤਿ ਮਹਤੁ']:
            with self.subTest(text=text):
                self.send(text,start=10)
                self.assertIsNone(server._live_match)
                self.assertTrue(server.live_state(self.sid,10)['waiting_for_shabad'])
        self.send('ਦਿਵਸੁ ਰਾਤਿ ਦੁਇ ਦਾਈ ਦਾਇਆ ਖੇਲੈ ਸਗਲ ਜਗਤੁ',start=30)
        self.assertIsNone(server._live_match)
        self.send('ਕਾਟੇ ਸੁ ਪਾਪ ਨਰਹੁ ਕੇ ਗੁਰ ਰਾਮਦਾਸ ਜਿਨ ਪਾਇਓ',start=50)
        self.assertEqual(server._live_match['pankti_id'],'BKS1')
        self.assertFalse(server.live_state(self.sid,50)['waiting_for_shabad'])

    def test_single_name_and_common_ardas_pairs_do_not_trigger_waiting(self):
        self.assertIsNone(server._non_shabad_span('ਵਾਹਿਗੁਰੂ ਜੀ ਸਹਾਇ'))
        self.assertIsNone(server._non_shabad_span('ਗੁਰ ਨਾਨਕ'))
        self.assertIsNone(server._non_shabad_span('ਗੁਰੂ ਪਾਣੀ'))
        for text in ['ਵਡੇ ਹੂ ਵਡਾ ਅਪਾਰ', 'ਪਵਨ ਗੁਰੂ ਪਾਣੀ ਪਿਤਾ']:
            self.assertEqual(server._non_shabad_span(text)[2],'role-ending')

    def test_reconnect_offsets_are_applied_to_special_waiting(self):
        server._pankti_timeline[:]=[(1060,dict(server._live_match))]
        self.send('ਵਾਹਿਗੁਰੂ ਵਾਹਿਗੁਰੂ',start=2,offset=60)
        self.assertFalse(server.live_state(self.sid,61.9)['waiting_for_shabad'])
        self.assertEqual(server.live_state(self.sid,62)['waiting_reason'],'simran')
