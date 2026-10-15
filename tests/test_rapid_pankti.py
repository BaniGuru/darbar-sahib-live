import asyncio
import json
import time
import unittest
from unittest.mock import patch

import server
from bani_matching import OrderedBaniMatcher
from pankti_display import display_lines


class RapidPanktiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if server._pankti_rows is None:
            server._build_pankti_index()
        cls.rows={r['pankti_id']:r for r in server._pankti_rows}

    def setUp(self):
        initial=server._active_match_payload(self.rows['N43J'],1,2)
        p=patch.multiple(server,_bani_matcher=None,_active_shabad_id='NJM',_live_match=initial,
                        _live_match_at=time.time(),_pending_shabad_id=None,_pending_shabad_hits=0,
                        _pending_shabad_update=None,_pending_pankti_id=None,_pending_pankti_hits=0,
                        _pending_pankti_audio_time=None,_debug_final_text='',_debug_partial_text='',
                        _debug_timed_final_words=[],_debug_timed_partial_words=[],_debug_recompute_from=None,
                        _pankti_timeline=[],_pankti_display_timeline=[],_display_dirty_from=None,
                        _last_greeting_key=None,_greeting_audio_cutoff=None,_stream_wall_started_at=1000,
                        _audio_sessions={},_audio_subscribers=set())
        p.start();self.addCleanup(p.stop)

    def send(self,text,start=20,spacing=.25):
        words=[dict(text=word,start=start+i*spacing,end=start+(i+1)*spacing)
               for i,word in enumerate(text.split())]
        message=dict(type='partial',partial_text=text,partial_words=words,
                     committed_words=[],committed_text='',window_end=words[-1]['end'],audio_active=True)
        class Socket:
            async def __aiter__(self):
                yield json.dumps(message,ensure_ascii=False)
        with patch.object(server.logger,'info'):
            asyncio.run(server._handle_asr_messages(Socket()))
        return words

    def test_two_words_advance_without_second_asr_update(self):
        words=self.rows['ERNS']['_tokens'][:2]
        self.send(' '.join(words))
        self.assertEqual(server._live_match['pankti_id'],'ERNS')
        self.assertEqual(server._pankti_timeline[-1][0],1020)
        self.assertEqual([w['pankti_id'] for w in server._debug_timed_partial_words],['ERNS']*2)

    def test_two_words_can_follow_a_sung_backward_repeat(self):
        server._live_match=server._active_match_payload(self.rows['BKS1'],1,2)
        self.send(' '.join(self.rows['ERNS']['_tokens'][:2]))
        self.assertEqual(server._live_match['pankti_id'],'ERNS')

    def test_fast_batch_preserves_all_intermediate_lines(self):
        ids=['ERNS','9E67','6KLH']
        text=' '.join(' '.join(self.rows[pid]['_tokens']) for pid in ids)
        words=self.send(text,spacing=.12)
        detected=[line['pankti_id'] for _,line in server._pankti_timeline]
        for pid in ids:
            self.assertIn(pid,detected)
        self.assertEqual(server._live_match['pankti_id'],ids[-1])
        start=0
        for pid in ids:
            at=1000+words[start]['start']
            self.assertTrue(any(line['pankti_id']==pid and abs(ts-at)<.001
                                for ts,line in server._pankti_timeline))
            start+=len(self.rows[pid]['_tokens'])

    def test_future_pair_is_prepared_once_and_expires_at_its_boundary(self):
        first=server._active_match_payload(self.rows['ERNS'],1,2)
        second=server._active_match_payload(self.rows['9E67'],1,2)
        server._pankti_timeline[:]=[(1020,first)]
        server._display_dirty_from=1014
        server._prepare_pankti_timeline()
        self.assertEqual(len(server._prepared_display(1020,first)),1)
        server._pankti_timeline.append((1022,second))
        server._display_dirty_from=1016
        server._prepare_pankti_timeline()
        self.assertEqual([p['pankti_id'] for p in server._prepared_display(1020,first)],['ERNS','9E67'])
        self.assertEqual([p['pankti_id'] for p in server._prepared_display(1022,second)],['ERNS','9E67'])
        self.assertEqual([p['pankti_id'] for p in server._prepared_display(1028,second)],['9E67'])

    def test_unchanged_snapshots_do_not_run_matching_again(self):
        text=' '.join(self.rows['ERNS']['_tokens'][:2])
        message=dict(type='partial',partial_text=text,partial_words=[
            dict(text=w,start=20+i,end=20+i+.8) for i,w in enumerate(text.split())],
            committed_words=[],committed_text='',window_end=24,audio_active=True)
        class Socket:
            async def __aiter__(self):
                for _ in range(5):
                    yield json.dumps(message,ensure_ascii=False)
        with patch.object(server,'_match_asr_prefix',wraps=server._match_asr_prefix) as matching, \
             patch.object(server.logger,'info'):
            asyncio.run(server._handle_asr_messages(Socket()))
        self.assertEqual(matching.call_count,2)
        self.assertEqual(server._live_match['pankti_id'],'ERNS')

    def test_old_alignment_does_not_consume_unmatched_new_words(self):
        _,_,end=server._ordered_alignment_score(['ਨਾਮ','ਜਪਹ','ਹਰ'],['ਨਾਮ','ਜਪਹ','ਹਰ','ਸੁਰ','ਦੇਵ'])
        self.assertEqual(end,3)

    def test_refrain_wrap_does_not_jump_to_another_lines_middle(self):
        server._active_shabad_id='GDB'
        server._live_match=server._active_match_payload(self.rows['8J7L'],1,2)
        self.send('ਜਪਹੁ ਹਰਿ ਨਾਮੁ ਜਪਹੁ ਹਰਿ ਨਾਮੁ')
        self.assertEqual(server._live_match['pankti_id'],'8J7L')
        self.assertNotIn('P0X8',[line['pankti_id'] for _,line in server._pankti_timeline])

    def test_two_word_bani_line_is_recorded_before_auto_next(self):
        rows=[]
        for pid,text in [('first','ਸਚੁ ਨਾਮੁ'),('second','ਗੁਰ ਕਿਰਪਾ')]:
            rows.append(dict(pankti_id=pid,shabad_id='custom',order_id=len(rows),line_group=1,
                             _tokens=server._normalise_query(text).split(),punjabi=text,legacy_gurmukhi=text,english=''))
        matcher=OrderedBaniMatcher({13:dict(family='asa-ki-vaar',name='Asa Ki Vaar',lines=rows)},
                                  server._token_similarity,server._normalise_query)
        matcher.family='asa-ki-vaar';matcher.positions={13:0};matcher.selected=13
        server._bani_matcher=matcher;server._active_shabad_id='custom'
        server._live_match=server._active_match_payload(rows[0],1,2)
        with patch.object(server,'match_live_pankti',return_value=None):
            self.send('ਸਚੁ ਨਾਮੁ')
        self.assertEqual([(at,line['pankti_id']) for at,line in server._pankti_timeline],
                         [(1020,'first'),(1020.5,'second')])
        self.assertEqual([w['pankti_id'] for w in server._debug_timed_partial_words],['first','first'])

    def test_poll_reads_prepared_ids_without_matching_or_pairing(self):
        self.send(' '.join(self.rows['ERNS']['_tokens'][:2]))
        sid='11111111-1111-4111-8111-111111111111'
        server._audio_sessions[sid]=dict(last_seen=time.monotonic(),first_audio_at=1020,generation=1)
        with patch.object(server,'match_live_pankti',side_effect=AssertionError('poll matched')), \
             patch.object(server,'display_lines',side_effect=AssertionError('poll paired')):
            first=server.live_state(sid,.3)
            later=server.live_state(sid,.4)
        self.assertEqual(first['pankti_id'],'ERNS')
        self.assertEqual(first['display_panktis'][0]['pankti_id'],'ERNS')
        self.assertEqual([w['pankti_id'] for w in later['playback_words']],['ERNS']*2)


class PairDisplayTests(unittest.TestCase):
    @staticmethod
    def line(pid,shabad='s',**extra):
        return dict(pankti_id=pid,shabad_id=shabad,punjabi=pid,legacy_gurmukhi=pid,english=pid,**extra)

    def test_short_forward_adjacent_lines_remain_in_reading_order(self):
        first,second=self.line('1'),self.line('2')
        timeline=[(10,first),(12,second)]
        for playhead,selected in [(10,first),(12,second),(17,second)]:
            self.assertEqual([p['pankti_id'] for p in display_lines(timeline,playhead,selected,{'1':'2'})],['1','2'])
        self.assertEqual(len(display_lines(timeline,18,second,{'1':'2'})),1)

    def test_long_gap_skipped_line_backward_jump_and_other_shabad_stay_single(self):
        for first,second,gap in [(self.line('1'),self.line('2'),7),
                                 (self.line('1'),self.line('3'),2),
                                 (self.line('4'),self.line('1'),2),
                                 (self.line('1'),self.line('2','other'),2)]:
            result=display_lines([(10,first),(10+gap,second)],10+gap,second,{'1':'2','2':'3'})
            self.assertEqual([p['pankti_id'] for p in result],[second['pankti_id']])

    def test_held_notes_in_first_line_do_not_prevent_a_close_transition(self):
        first=self.line('1',last_word_timestamp=25)
        second=self.line('2')
        timeline=[(10,first),(30,second)]
        self.assertEqual(len(display_lines(timeline,20,first,{'1':'2'})),1)
        self.assertEqual([p['pankti_id'] for p in display_lines(timeline,25,first,{'1':'2'})],['1','2'])
        self.assertEqual(len(display_lines(timeline,31,second,{'1':'2'})),2)
        distant=self.line('1',last_word_timestamp=20)
        self.assertEqual(len(display_lines([(10,distant),(30,second)],31,second,{'1':'2'})),1)

    def test_prediction_alone_cannot_create_a_pair(self):
        first,second=self.line('1'),self.line('2',match_mode='bani-auto-next')
        self.assertEqual(len(display_lines([(10,first),(12,second)],12,second,{'1':'2'})),1)

    def test_bani_pairs_require_adjacent_occurrences_on_the_same_route(self):
        first=self.line('1',bani_id=13,bani_position=5)
        second=self.line('2',bani_id=13,bani_position=6)
        self.assertEqual(len(display_lines([(10,first),(12,second)],12,second,{'1':'2'})),2)
        backward=self.line('4',bani_id=13,bani_position=5)
        restart=self.line('1',bani_id=13,bani_position=6)
        self.assertEqual(len(display_lines([(10,backward),(12,restart)],12,restart,{'1':'2'})),1)
        wrong=self.line('2',bani_id=13,bani_position=9)
        self.assertEqual(len(display_lines([(10,first),(12,wrong)],12,wrong,{})),1)
