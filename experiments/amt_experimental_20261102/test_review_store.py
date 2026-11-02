"""Verify new annotations cannot cross cohorts and survive server restarts."""
import hashlib
import json
import tempfile
import unittest
import re
from pathlib import Path
from common import RUN_ID, save, sha
from review_server import create_app, PREFIX


class ReviewStoreTest(unittest.TestCase):
    def test_bound_persistent_annotations_and_authorization(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);image=root/'scenes/scene0012/review/media/test.gif'
            image.parent.mkdir(parents=True);image.write_bytes(b'GIF89a')
            records=[{'candidate_id':f'12:1:2:{i}','scene_num':12,'review_rank':i+1,
                      'candidate_frame_raw':i,'score':1-i/100,'vanished_track_id':1,'survivor_track_id':2,
                      'asset':str(image.relative_to(root)),'asset_sha256':sha(image),'tracks':[]} for i in range(80)]
            path=root/'scenes/scene0012/review/manifest.json'
            document={'run_id':RUN_ID,'budget':80,'smoke_only':False,'candidates':records,
                      'model_sha256':'fixed-model','prediction_sha256':'fixed-predictions'}
            save(path,document)
            access={'key':'test-access','session_secret':'test-session-secret'}
            app=create_app(root,access);app.testing=True;client=app.test_client()
            base='https://review.example'
            self.assertEqual(client.get(PREFIX+'/api/status',base_url=base).status_code,403)
            self.assertEqual(client.get(PREFIX+'/?key=test-access',base_url=base).status_code,302)
            page=client.get(PREFIX+'/',base_url=base);self.assertEqual(page.status_code,200)
            token=re.search(r'name="csrf-token" content="([^"]+)"',page.data.decode()).group(1)
            route=PREFIX+'/api/scenes/12/events/12:1:2:0/review'
            self.assertEqual(client.post(route,json={'label':'collision'},base_url=base).status_code,403)
            headers={'X-CSRF-Token':token}
            self.assertEqual(client.post(route,json={'label':'invalid'},headers=headers,base_url=base).status_code,400)
            self.assertEqual(client.post(route,json={'label':'collision','note':'test evidence'},headers=headers,base_url=base).status_code,200)
            self.assertEqual(client.post(PREFIX+'/api/scenes/12/events/not-in-top80/review',json={'label':'collision'},headers=headers,base_url=base).status_code,404)
            restarted=create_app(root,access);restarted.testing=True;other=restarted.test_client()
            other.get(PREFIX+'/?key=test-access',base_url=base)
            status=other.get(PREFIX+'/api/status',base_url=base).get_json()
            self.assertEqual(status['scenes'][0]['reviewed'],1)
            export=other.get(PREFIX+'/export.csv',base_url=base).data.decode('utf-8-sig')
            self.assertIn('fixed-model,fixed-predictions',export);self.assertIn('test evidence',export)
            self.assertEqual(other.get(PREFIX+'/assets/../../reviews.sqlite3',base_url=base).status_code,404)
            token2=re.search(r'name="csrf-token" content="([^"]+)"',other.get(PREFIX+'/',base_url=base).data.decode()).group(1)
            undo=PREFIX+'/api/scenes/12/events/12:1:2:0/undo'
            self.assertEqual(other.post(undo,json={},base_url=base).status_code,403)
            self.assertEqual(other.post(undo,json={},headers={'X-CSRF-Token':token2},base_url=base).status_code,200)
            self.assertEqual(other.get(PREFIX+'/api/status',base_url=base).get_json()['scenes'][0]['reviewed'],0)
            import sqlite3
            with sqlite3.connect(root/'reviews.sqlite3') as db:
                self.assertEqual(db.execute('SELECT label FROM history ORDER BY id').fetchall(),[('collision',),('unreviewed',)])
            self.assertTrue(image.exists())
            with other.get(PREFIX+'/api/scenes/12/events/12:1:2:0/movie',base_url=base) as response:
                self.assertEqual(response.data,b'GIF89a')
            document['prediction_sha256']='different-predictions';save(path,document)
            with self.assertRaisesRegex(RuntimeError,'Frozen cohort changed'):
                other.get(PREFIX+'/api/scenes/12',base_url=base)


if __name__=='__main__':unittest.main()
