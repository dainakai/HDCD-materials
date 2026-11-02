"""Progress must distinguish final measurements, remote import and stale data."""
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch
from monitor_snapshot import last_frame,now,scene_snapshot
from monitor_server import Collector,create_app,PREFIX


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


class MonitorTest(unittest.TestCase):
    def test_measurement_progress_uses_particle_output_not_finished_detection(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);folder=root/'scenes/scene0015/measurement';folder.mkdir(parents=True)
            (folder/'frame_stats.csv').write_text('frame,file\n10916,010917.png\n')
            (folder/'particles_3d.csv').write_text('frame,file\n49,000050.png\n50,partial')
            result=scene_snapshot(root,15,'measurement')
            self.assertEqual(result['progress']['done'],50)
            self.assertEqual(result['progress']['total'],10917)
            self.assertFalse(result['review_ready'])

    def test_remote_outputs_are_not_reviewable_before_local_import(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);receipt=root/'distributed/kit-scene0015.json'
            save(receipt,{'status':'running','stage':'transfer_results'})
            save(root/'campaign-state.json',{'status':'running','pid':1,'updated_at':now(),'workers':[
                {'worker':'daiPCatKIT-gpu0','status':'running','scene':15,'stage':'transfer_results'}]})
            save(root/'distributed/assignments.json',{'daiPCatKIT-gpu0':{'status':'running','scene':15,'pid':2,'receipt':str(receipt),'started_at':now()}})
            remote={'observed_at':now(),'hardware':{'gpus':[]},'scenes':[{'scene':15,'stage':None,'status':'PASS',
                'process_alive':False,'review_ready':True,'completed_stages':['background','minip','measurement','tracking','inference','assets'],'progress':None}]}
            collector=Collector(root,lambda scenes:remote)
            with patch('monitor_server.alive',return_value=True),patch('monitor_server.hardware',return_value={'gpus':[],'observed_at':now()}):
                value=collector.collect()
            self.assertEqual(value['reviewable'],0)
            self.assertEqual(value['scenes'][2]['completed_stages'],remote['scenes'][0]['completed_stages'])
            collector.remote=lambda scenes:(_ for _ in ()).throw(OSError('private host details'))
            with patch('monitor_server.alive',return_value=True),patch('monitor_server.hardware',return_value={'gpus':[],'observed_at':now()}):
                value=collector.collect()
            self.assertEqual(value['workers'][0]['status'],'connection_lost')
            self.assertEqual(value['workers'][0]['telemetry_at'],remote['observed_at'])
            self.assertNotIn('private host details',json.dumps(value))

    def test_http_serves_only_monitor_resources(self):
        collector=Collector();collector.latest={'schema':1,'status':'fixture'}
        app=create_app(collector);client=app.test_client()
        with client.get(PREFIX+'/') as response:self.assertEqual(response.status_code,200)
        self.assertEqual(client.get(PREFIX+'/api/state').get_json()['schema'],1)
        with client.get(PREFIX+'/assets/monitor.js') as response:self.assertEqual(response.status_code,200)
        self.assertEqual(client.get(PREFIX+'/assets/reviews.sqlite3').status_code,404)
        self.assertEqual(client.post(PREFIX+'/api/state',json={}).status_code,405)


if __name__=='__main__':unittest.main()
