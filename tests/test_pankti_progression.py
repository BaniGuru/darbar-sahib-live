import asyncio
import json
import time
import unittest
from unittest.mock import patch

import server


class PanktiProgressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if server._pankti_rows is None:
            server._build_pankti_index()
        cls.rows = {r['pankti_id']: r for r in server._pankti_rows}

    def setUp(self):
        values = dict(_bani_matcher=None, _active_shabad_id='SAM',
                      _live_match=server._active_match_payload(self.rows['99AN'],1,12),
                      _live_match_at=time.time(), _pending_shabad_id=None,
                      _pending_shabad_hits=0, _pending_pankti_id=None,
                      _pending_pankti_hits=0, _pending_pankti_audio_time=None,
                      _debug_final_text='', _debug_partial_text='',
                      _debug_timed_final_words=[], _debug_timed_partial_words=[],
                      _debug_recompute_from=None, _pankti_timeline=[],
                      _stream_wall_started_at=1000, _last_greeting_key=None,
                      _greeting_audio_cutoff=None, _audio_sessions={},
                      _audio_subscribers=set())
        p=patch.multiple(server,**values);p.start();self.addCleanup(p.stop)

    def run_messages(self, messages):
        observed=[]
        class Socket:
            async def __aiter__(self):
                for message in messages:
                    yield json.dumps(message,ensure_ascii=False)
                    observed.append(dict(server._live_match or {}))
        with patch.object(server.logger,'info'):
            asyncio.run(server._handle_asr_messages(Socket()))
        return observed

    @staticmethod
    def message(text, start=0):
        words=[dict(text=w,start=start+i,end=start+i+.8) for i,w in enumerate(text.split())]
        return dict(type='partial',partial_text=text,partial_words=words,
                    window_end=start+len(words),audio_active=True)

    def test_recorded_shared_pair_cannot_lock_previous_shabad(self):
        text='ਕਾਟੇ ਸੁ ਪਾਪ ਨਰਹੁ ਕੇ ਗੁਰ ਰਾਮਦਾਸ ਜਿਨ ਪਾਇਓ'
        observed=self.run_messages([self.message(text,20)]*2)
        self.assertEqual(observed[0]['pankti_id'],'99AN')
        self.assertEqual(observed[1]['pankti_id'],'BKS1')
        self.assertEqual(server._active_shabad_id,'NJM')
        self.assertEqual(server._pankti_timeline[-1][1]['pankti_id'],'BKS1')
        self.assertGreaterEqual(server._pankti_timeline[-1][0],1020)
        self.assertLessEqual(server._pankti_timeline[-1][0],1029)

    def test_shared_names_alone_do_not_switch_shabad(self):
        heard='ਗੁਰ ਰਾਮਦਾਸ ਗੁਰ ਰਾਮਦਾਸ'
        active=server.match_live_pankti(heard,'SAM',.55)
        target={**server._active_match_payload(self.rows['BKS1'],.9,4),'_matched_words':3}
        self.assertFalse(server._stronger_new_phrase(target,active,heard))
        self.run_messages([self.message(heard,20)]*3)
        self.assertEqual(server._active_shabad_id,'SAM')

    def test_distinctive_same_shabad_line_can_override_shared_pair(self):
        text='ਕਾਟੇ ਸੁ ਪਾਪ ਨਰਹੁ ਕੇ ਗੁਰ ਰਾਮਦਾਸ ਜਿਨ ਪਾਇਓ'
        active=server.match_live_pankti(text,'SAM',.55)
        target={**server.match_live_pankti(text,None,.82), 'shabad_id':'SAM'}
        self.assertTrue(server._stronger_new_phrase(target,active,text))

    def test_complete_greeting_fuzzy_and_not_name_refrain(self):
        self.assertEqual(server._greeting_end('ਵਾਹਿਗੁਰੂ ਜੀ ਕਾ ਖ਼ਾਲਸਾ ਵਾਹਿਗੁਰੂ ਜੀ ਕੀ ਫ਼ਤਹਿ'),8)
        self.assertEqual(server._greeting_end('ਵਾਹਿਗੁਰੂ ਜੀ ਕਾ ਖਾਲਸਾ ਵਾਹਗੁਰੂ ਜੀ ਕੀ ਫਤਿਹ'),8)
        self.assertIsNone(server._greeting_end('ਵਾਹਿਗੁਰੂ ਵਾਹਿਗੁਰੂ ਵਾਹਿਗੁਰੂ ਵਾਹਿਗੁਰੂ'))
        self.assertIsNone(server._greeting_end('ਵਾਹਿਗੁਰੂ ਜੀ ਕਾ ਖਾਲਸਾ'))
        self.assertEqual(server._greeting_end('ASR 42 ਵਾਹਿਗੁਰੂ ਜੀ ਕਾ ਖਾਲਸਾ ਵਾਹਿਗੁਰੂ ਜੀ ਕੀ ਫਤਹਿ ਨਾਮ ਜਪੋ'),10)

    def test_greeting_waits_on_playback_timeline_and_only_fresh_shabad_exits(self):
        sid='11111111-1111-4111-8111-111111111111'
        old=server._live_match.copy()
        server._pankti_timeline.append((1000,old))
        server._audio_sessions[sid]=dict(last_seen=time.monotonic(),first_audio_at=1000,generation=1)
        greeting='ਵਾਹਿਗੁਰੂ ਜੀ ਕਾ ਖ਼ਾਲਸਾ ਵਾਹਿਗੁਰੂ ਜੀ ਕੀ ਫ਼ਤਹਿ'
        self.run_messages([self.message(greeting,10)]*2)
        self.assertEqual(len(server._pankti_timeline),2)
        self.assertFalse(server.live_state(sid,17)['waiting_for_shabad'])
        self.assertTrue(server.live_state(sid,18)['waiting_for_shabad'])
        self.assertEqual(server.live_state(sid,18)['pankti'],'')
        self.assertTrue(server.live_state(sid,900)['waiting_for_shabad'])
        # Old timed words retained by the ASR stream are not new evidence.
        self.run_messages([self.message('ਕਾਟੇ ਸੁ ਪਾਪ ਨਰਹੁ ਕੇ ਗੁਰ ਰਾਮਦਾਸ ਜਿਨ ਪਾਇਓ',0)])
        self.assertIsNone(server._live_match)
        self.run_messages([self.message('ਕਾਟੇ ਸੁ ਪਾਪ ਨਰਹੁ ਕੇ ਗੁਰ ਰਾਮਦਾਸ ਜਿਨ ਪਾਇਓ',30)])
        self.assertEqual(server._live_match['pankti_id'],'BKS1')
        self.assertTrue(server.live_state(sid,20)['waiting_for_shabad'])
        self.assertFalse(server.live_state(sid,40)['waiting_for_shabad'])
        self.assertEqual(server.live_state(sid,40)['pankti_id'],'BKS1')
        self.assertFalse(server.live_state(sid,900)['waiting_for_shabad'])

    def test_repeated_phrase_timestamp_uses_latest_occurrence(self):
        text='ਗੁਰ ਰਾਮਦਾਸ ਨਾਮ ਜਪੋ ਗੁਰ ਰਾਮਦਾਸ'
        self.assertEqual(server._matched_word_offset(self.message(text,10),text,6,
                         'ਗੁਰ ਰਾਮਦਾਸ',query_start=4),14)
