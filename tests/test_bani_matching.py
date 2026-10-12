import asyncio
import json
import sqlite3
import unittest
from unittest.mock import AsyncMock, patch

import server
from bani_matching import OrderedBaniMatcher, bani_family


class BaniMatchingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        server._build_pankti_index()
        cls.routes = server._bani_matcher.routes
        cls.by_id = {row['pankti_id']: row for row in server._pankti_rows}

    def setUp(self):
        self.matcher = OrderedBaniMatcher(self.routes, server._token_similarity, server._normalise_query)
        self.records = []

    def heard(self, text):
        for word in text.split():
            start = len(self.records)
            self.records.append({'text': word, 'start': start, 'end': start + .8})
        return ' '.join(w['text'] for w in self.records), {
            'type': 'partial', 'committed_words': list(self.records), 'partial_words': [],
            'committed_text': ' '.join(w['text'] for w in self.records),
            'partial_text': '', 'window_end': len(self.records)}

    def adopt(self, row):
        text, message = self.heard(' '.join(self.matcher.spoken_tokens(row)))
        return self.matcher.adopt(server._active_match_payload(row, 1, 12), text, message)

    def test_database_order_is_group_then_lines_order_not_shabad_order(self):
        with sqlite3.connect(server.BANI_DB_PATH) as conn:
            for bani_id, route in self.routes.items():
                expected = [(str(pid), group) for pid, group in conn.execute('''
                    SELECT p.id, bl.line_group FROM banis b
                    JOIN bani_lines bl ON bl.bani_id=b.id
                    JOIN panktis p ON p.id=bl.line_id JOIN lines l ON l.id=p.id
                    WHERE b.id=? ORDER BY bl.line_group,l.order_id,p.id
                ''', (bani_id,)) if str(pid) in self.by_id]
                self.assertEqual([(r['pankti_id'],r['line_group']) for r in route['lines']], expected)

    def test_family_names_include_variants(self):
        for name in ['Aarti', 'Aarti (Longer)']: self.assertEqual(bani_family(name), 'aarti')
        for name in ['Rehras Sahib (S.)', 'Rehraas Sahib']: self.assertEqual(bani_family(name), 'rehras')
        for name in ['Asa Ki Var','Asa Ki Vaar']: self.assertEqual(bani_family(name), 'asa-ki-vaar')
        self.assertIsNone(bani_family('Jap Ji Sahib'))

    def test_shared_invocation_does_not_start_a_bani_session(self):
        row=next(r for r in self.routes[9]['lines'] if r.get('_line_type')==1)
        result=self.adopt(row)
        self.assertNotIn('bani_id',result)
        self.assertIsNone(self.matcher.family)

    def test_line_end_predicts_next_without_any_next_words(self):
        route = self.routes[13]['lines']
        result = self.adopt(route[1])
        self.assertEqual(result['pankti_id'], route[2]['pankti_id'])
        self.assertTrue(result['_bani_auto_next'])
        text, message = self.heard(self.matcher.spoken_tokens(route[2])[0])
        result = self.matcher.match(text, message)
        self.assertEqual(result['pankti_id'], route[2]['pankti_id'])
        self.assertFalse(result['_bani_auto_next'])

    def test_prediction_does_not_confirm_itself_on_unchanged_or_revised_end(self):
        result = self.adopt(self.routes[13]['lines'][1])
        position = dict(self.matcher.positions)
        evidence = dict(self.matcher.evidence)
        text, message = self.heard('')
        for _ in range(4): self.assertIsNone(self.matcher.match(text, message))
        message['committed_words'][-1]['end'] += .2
        self.assertIsNone(self.matcher.match(text,message))
        self.assertEqual(self.matcher.positions,position)
        self.assertEqual(dict(self.matcher.evidence),evidence)

    def test_line_end_and_next_first_word_in_one_update_preserves_that_word(self):
        route=self.routes[13]['lines']
        row=route[1]
        first=' '.join(self.matcher.spoken_tokens(route[2])[:3])
        text,message=self.heard(' '.join(self.matcher.spoken_tokens(row))+ ' '+first)
        result=self.matcher.adopt(server._active_match_payload(row,1,12),text,message)
        self.assertEqual(result['pankti_id'],route[2]['pankti_id'])
        self.assertTrue(result['_bani_auto_next'])
        confirmed=self.matcher.match(text,message)
        self.assertEqual(confirmed['pankti_id'],route[2]['pankti_id'])
        self.assertFalse(confirmed['_bani_auto_next'])

    def test_finished_line_repeat_requires_three_words_before_going_back(self):
        row=self.routes[13]['lines'][1]
        self.adopt(row)
        words=self.matcher.spoken_tokens(row)
        text,message=self.heard(' '.join(words[:2]))
        self.assertIsNone(self.matcher.match(text,message))
        text,message=self.heard(words[2])
        result=self.matcher.match(text,message)
        self.assertEqual(result['pankti_id'],row['pankti_id'])
        self.assertFalse(result['_bani_auto_next'])

    def test_rahao_is_not_required_to_finish_a_sung_line(self):
        row = next(r for r in self.routes[9]['lines'] if r['_tokens'][-1]=='ਰਹਾਉ')
        self.assertTrue(self.adopt(row)['_bani_auto_next'])

    def test_strong_terminal_bigram_survives_an_asr_error_before_it(self):
        row=self.routes[9]['lines'][3]
        words=self.matcher.spoken_tokens(row)
        words[-3]='ਗਲਤਸ਼ਬਦ'
        text,message=self.heard(' '.join(words))
        result=self.matcher.adopt(server._active_match_payload(row,1,12),text,message)
        self.assertTrue(result['_bani_auto_next'])

    def test_pauri_ending_tolerates_a_single_asr_consonant_substitution(self):
        row=self.by_id['N5Y1']
        words=self.matcher.spoken_tokens(row)
        words[-1]='ਜਾਉ'
        text,message=self.heard(' '.join(words))
        result=self.matcher.adopt(server._active_match_payload(row,1,12),text,message)
        self.assertTrue(result['_bani_auto_next'])
        self.assertNotEqual(result['pankti_id'],row['pankti_id'])

    def test_cross_group_advance_uses_bani_order(self):
        route = self.routes[13]['lines']
        p = next(i for i in range(len(route)-1) if route[i]['line_group']!=route[i+1]['line_group'])
        result = self.adopt(route[p])
        self.assertEqual(result['pankti_id'],route[p+1]['pankti_id'])
        self.assertEqual(result['line_group'],route[p+1]['line_group'])

    def test_long_aarti_exclusive_line_changes_variant_and_common_line_resumes(self):
        short,long = self.routes[9]['lines'],self.routes[10]['lines']
        short_ids = {r['pankti_id'] for r in short}
        p = next(i for i,r in enumerate(long) if r['pankti_id'] not in short_ids)
        self.adopt(long[p-1])
        text,message = self.heard(' '.join(self.matcher.spoken_tokens(long[p])))
        result = self.matcher.match(text,message)
        self.assertIsNotNone(result)
        self.assertEqual(self.matcher.selected,10)
        self.assertGreater(self.matcher.evidence[10],self.matcher.evidence[9])
        common = next(r for r in long[p+1:] if r['pankti_id'] in short_ids)
        self.adopt(common)
        self.assertEqual(self.matcher.selected,10)
        self.assertIn(9,self.matcher.positions)

    def test_all_target_routes_can_follow_complete_spoken_lines(self):
        for bani_id, route in self.routes.items():
            self.setUp()
            start = next(i for i,row in enumerate(route['lines']) if row.get('_line_type') != 1)
            first = route['lines'][start]
            self.adopt(first)
            for p,row in enumerate(route['lines'][start+1:],start+1):
                text,message = self.heard(' '.join(self.matcher.spoken_tokens(row)))
                result = self.matcher.match(text,message)
                self.assertIsNotNone(result,(bani_id,p,row['speech']))
                self.assertEqual(self.matcher.positions[bani_id],p,(bani_id,p,result))

    def test_outside_shabad_suspends_and_bani_can_resume(self):
        route=self.routes[13]['lines']
        self.adopt(route[1])
        outside=next(r for r in server._pankti_rows if r['pankti_id'] not in self.matcher.members)
        text,message=self.heard(' '.join(outside['_tokens']))
        result=self.matcher.adopt(server._active_match_payload(outside,1,12),text,message)
        self.assertEqual(result['pankti_id'],outside['pankti_id'])
        self.assertIsNone(self.matcher.family)
        self.adopt(route[2])
        self.assertEqual(self.matcher.family,'asa-ki-vaar')

    def test_aarti_shabad_inserted_in_asa_stays_an_ordinary_shabad(self):
        self.adopt(self.routes[13]['lines'][1])
        row=self.routes[9]['lines'][1]
        for _ in range(2):
            result=self.adopt(row)
            self.assertNotIn('bani_id',result)
            self.assertIsNone(self.matcher.family)
        self.adopt(self.routes[13]['lines'][2])
        self.assertEqual(self.matcher.family,'asa-ki-vaar')

    def test_repeated_line_id_in_a_later_group_retains_occurrence_position(self):
        rows=[]
        for pid,group,text in [('A',1,'ਸਚੁ ਨਾਮੁ ਵਡਿਆਈ'),('B',1,'ਗੁਰ ਕੀ ਸੇਵਾ ਕਰਿ'),
                               ('A',2,'ਸਚੁ ਨਾਮੁ ਵਡਿਆਈ'),('C',2,'ਹਰਿ ਹਰਿ ਜਪੁ ਜੀਵੈ')]:
            rows.append({'pankti_id':pid,'shabad_id':'test','order_id':group,
                         'line_group':group,'_tokens':text.split(),'speech':text,
                         'punjabi':text,'legacy_gurmukhi':'','english':''})
        self.matcher=OrderedBaniMatcher({13:{'family':'asa-ki-vaar','name':'Asa Ki Vaar','lines':rows}},
                                        server._token_similarity,server._normalise_query)
        self.adopt(rows[0])
        for p in [1,2]:
            text,message=self.heard(' '.join(rows[p]['_tokens']))
            result=self.matcher.match(text,message)
            self.assertEqual(self.matcher.positions[13],p)
        self.assertEqual(result['pankti_id'],'C')
        self.assertEqual(result['line_group'],2)

    def test_short_aarti_end_does_not_predict_from_stale_long_variant(self):
        self.adopt(self.routes[9]['lines'][1])
        last=self.routes[9]['lines'][-1]
        result=self.adopt(last)
        self.assertEqual(result['pankti_id'],last['pankti_id'])
        self.assertFalse(result['_bani_auto_next'])

    def test_handler_auto_next_and_existing_two_hit_shabad_switch(self):
        route=self.routes[13]['lines']
        outside=next(r for r in server._pankti_rows if 'ਜੋਗੁ ਬਨਿਆ ਤੇਰਾ' in r['speech'])
        self.assertNotIn(outside['pankti_id'],self.matcher.members)
        messages=[]
        for row in [route[1],outside]:
            text,message=self.heard(' '.join(self.matcher.spoken_tokens(row)))
            messages.append(json.dumps(message,ensure_ascii=False))
        messages.append(messages[-1])
        observed=[]
        class Socket:
            async def __aiter__(sock):
                for message in messages:
                    yield message
                    observed.append(dict(server._live_match or {}))
        values={'_bani_matcher':self.matcher,'_active_shabad_id':None,'_live_match':None,
                '_live_match_at':0,'_pending_shabad_id':None,'_pending_shabad_hits':0,
                '_pending_pankti_id':None,'_pending_pankti_hits':0,'_pending_pankti_audio_time':None,
                '_debug_final_text':'','_debug_partial_text':'','_debug_timed_final_words':[],
                '_debug_recompute_from':None,'_pankti_timeline':[],'_stream_wall_started_at':100000}
        with patch.multiple(server,**values),patch.object(server.logger,'info'):
            asyncio.run(server._handle_asr_messages(Socket()))
        self.assertEqual(observed[0]['pankti_id'],route[2]['pankti_id'])
        self.assertEqual(observed[0]['match_mode'],'bani-auto-next')
        self.assertEqual(observed[1]['pankti_id'],route[2]['pankti_id'])
        self.assertEqual(observed[2]['shabad_id'],outside['shabad_id'])
        self.assertNotIn('bani_id',observed[2])

    def test_idle_audio_does_not_renew_stale_pankti_match(self):
        message = {'type': 'partial', 'audio_active': False,
                   'committed_text': 'ਹਰਿ ਕਾ ਨਾਮੁ', 'partial_text': '', 'window_end': 20}
        class Socket:
            async def __aiter__(self):
                yield json.dumps(message, ensure_ascii=False)
        values = {'_live_match_at': 42, '_live_match': {'pankti_id': 'old'},
                  '_debug_final_text': '', '_debug_partial_text': '',
                  '_debug_final_words': [], '_debug_partial_words': []}
        with patch.multiple(server, **values), \
             patch.object(server, '_broadcast_debug_words', new=AsyncMock()), \
             patch.object(server, 'match_live_pankti') as matcher:
            asyncio.run(server._handle_asr_messages(Socket()))
            matcher.assert_not_called()
            self.assertEqual(server._live_match_at, 42)
            self.assertEqual(server._debug_final_text, 'ਹਰਿ ਕਾ ਨਾਮੁ')


if __name__ == '__main__':
    unittest.main()
