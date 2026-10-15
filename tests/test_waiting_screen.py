import asyncio
from datetime import datetime
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
from zoneinfo import ZoneInfo

import server
from duty_schedule import DutySchedule


class DutyLookupTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name)/'duties.json'
        self.lookup = DutySchedule(self.path)

    def at(self, clock, date='15-10-2026'):
        return self.lookup.at(datetime.strptime(date+' '+clock,'%d-%m-%Y %H:%M').replace(tzinfo=ZoneInfo('Asia/Kolkata')).timestamp())

    def write(self, entries, date='15-10-2026'):
        self.path.write_text(json.dumps(dict(duties={date:entries},updated_at='test'),ensure_ascii=False))

    def test_missing_invalid_and_missing_date_or_time_have_no_name(self):
        self.assertIsNone(self.at('09:00'))
        self.path.write_text('broken JSON');self.assertIsNone(self.at('09:00'))
        self.write([dict(start_time='10:00',end_time='11:00',raagi='ਭਾਈ ਸਾਹਿਬ')])
        self.assertIsNone(self.at('09:00'));self.assertIsNone(self.at('10:00','16-10-2026'))

    def test_boundaries_and_overnight_do_not_reuse_previous_dates(self):
        self.write([dict(start_time='09:00',end_time='10:00',raagi='ਪਹਿਲੇ'),
                    dict(start_time='10:00',end_time='11:00',raagi='ਦੂਜੇ'),
                    dict(start_time='20:45',end_time='',raagi='ਅਖੀਰਲੇ')])
        self.assertEqual(self.at('09:59'),'ਪਹਿਲੇ');self.assertEqual(self.at('10:00'),'ਦੂਜੇ')
        self.assertIsNone(self.at('11:00'));self.assertEqual(self.at('23:59'),'ਅਖੀਰਲੇ')
        self.assertIsNone(self.at('00:00','16-10-2026'))

    def test_atomic_replacement_and_deletion_refresh_without_server_restart(self):
        self.write([dict(start_time='09:00',end_time='11:00',raagi='ਪਹਿਲੇ')])
        self.assertEqual(self.at('10:00'),'ਪਹਿਲੇ')
        replacement=self.path.with_suffix('.new')
        replacement.write_text(json.dumps(dict(duties={'15-10-2026':[dict(start_time='09:00',end_time='11:00',raagi='ਦੂਜੇ')]},updated_at='new')))
        replacement.replace(self.path)
        self.assertEqual(self.at('10:00'),'ਦੂਜੇ')
        self.path.unlink();self.assertIsNone(self.at('10:00'))

    def test_overlapping_or_malformed_slots_do_not_guess_a_name(self):
        for entries in [[dict(start_time='09:00',end_time='11:00',raagi='ਪਹਿਲੇ'),dict(start_time='10:00',end_time='12:00',raagi='ਦੂਜੇ')],
                        [dict(start_time='99:00',end_time='',raagi='ਗਲਤ')]]:
            self.write(entries);self.assertIsNone(self.at('10:00'))


class WaitingScreenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        server._build_pankti_index()
        cls.rows={r['pankti_id']:r for r in server._pankti_rows}

    def setUp(self):
        values=dict(_bani_matcher=None,_active_shabad_id='NJM',
                    _live_match=server._active_match_payload(self.rows['N43J'],1,2),_live_match_at=time.time(),
                    _pending_shabad_id=None,_pending_shabad_hits=0,_pending_shabad_update=None,
                    _pending_pankti_id=None,_pending_pankti_hits=0,_pending_pankti_audio_time=None,
                    _debug_final_text='',_debug_partial_text='',_debug_timed_final_words=[],_debug_timed_partial_words=[],
                    _debug_recompute_from=None,_pankti_timeline=[],_playback_pankti_timeline=[],
                    _stream_wall_started_at=1000,_greeting_audio_cutoff=None,_last_greeting_key=None,
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

    def test_timeout_is_30_seconds_from_latest_supported_word(self):
        old={k:v for k,v in server._live_match.items() if not k.startswith('_')}
        server._pankti_timeline[:]=[(1000,{**old,'matched_until':1005})]
        server._prepare_pankti_timeline()
        self.assertFalse(server.live_state(self.sid,34.999)['waiting_for_shabad'])
        self.assertEqual(server.live_state(self.sid,35)['waiting_reason'],'no-shabad')

    def test_never_matched_stream_times_out_and_exposes_future_boundary(self):
        server._prepare_pankti_timeline()
        self.assertFalse(server.live_state(self.sid,29.999)['waiting_for_shabad'])
        early=server.live_state(self.sid,25)
        self.assertEqual(early['pankti_schedule'][0]['audio_timestamp'],1030)
        self.assertEqual(server.live_state(self.sid,30)['waiting_reason'],'no-shabad')

    def test_idle_asr_cannot_keep_an_old_pankti_alive(self):
        old=dict(server._live_match)
        server._pankti_timeline[:]=[(1000,{**old,'matched_until':1005})]
        self.send(self.rows['N43J']['speech'],start=20,audio_active=False)
        self.assertFalse(server.live_state(self.sid,34.99)['waiting_for_shabad'])
        self.assertEqual(server.live_state(self.sid,35)['waiting_reason'],'no-shabad')

    def test_duty_switch_changes_state_revision_without_new_asr(self):
        server._prepare_pankti_timeline()
        with patch.object(server._duty_schedule,'at',return_value='ਪਹਿਲੇ'):
            first=server.live_state(self.sid,40)
        with patch.object(server._duty_schedule,'at',return_value='ਦੂਜੇ'):
            following=server.live_state(self.sid,40)
        self.assertNotEqual(first['revision'],following['revision'])
        self.assertEqual(following['raagi'],'ਦੂਜੇ')

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

    def test_simran_does_not_consume_a_later_shabad_in_same_batch(self):
        self.send('ਵਾਹਿਗੁਰੂ ਵਾਹਿਗੁਰੂ ਕਾਟੇ ਸੁ ਪਾਪ ਨਰਹੁ ਕੇ ਗੁਰ ਰਾਮਦਾਸ ਜਿਨ ਪਾਇਓ')
        self.assertEqual(server._live_match['pankti_id'],'BKS1')
        self.assertTrue(server.live_state(self.sid,11)['waiting_for_shabad'])
        self.assertFalse(server.live_state(self.sid,20)['waiting_for_shabad'])

    def test_special_waiting_upgrades_after_30_seconds_and_falls_back_without_duty(self):
        self.send('ਵਾਹਿਗੁਰੂ ਵਾਹਿਗੁਰੂ')
        with patch.object(server._duty_schedule,'at',return_value='ਭਾਈ ਸਾਹਿਬ'):
            self.assertIsNone(server.live_state(self.sid,39.9)['raagi'])
            timed=server.live_state(self.sid,40)
            self.assertEqual(timed['waiting_reason'],'no-shabad');self.assertEqual(timed['raagi'],'ਭਾਈ ਸਾਹਿਬ')
        with patch.object(server._duty_schedule,'at',return_value=None):
            self.assertIsNone(server.live_state(self.sid,40)['raagi'])

    def test_single_name_and_common_ardas_pairs_do_not_trigger_waiting(self):
        self.assertIsNone(server._non_shabad_span('ਵਾਹਿਗੁਰੂ ਜੀ ਸਹਾਇ'))
        self.assertIsNone(server._non_shabad_span('ਗੁਰ ਨਾਨਕ'))

    def test_reconnect_offsets_are_applied_to_special_waiting(self):
        server._pankti_timeline[:]=[(1060,dict(server._live_match))]
        self.send('ਵਾਹਿਗੁਰੂ ਵਾਹਿਗੁਰੂ',start=2,offset=60)
        self.assertFalse(server.live_state(self.sid,61.9)['waiting_for_shabad'])
        self.assertEqual(server.live_state(self.sid,62)['waiting_reason'],'simran')
