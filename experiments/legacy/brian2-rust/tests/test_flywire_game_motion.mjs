import test from 'node:test';
import assert from 'node:assert/strict';
import {approachMotion, landingPosition, descentDuration} from '../validation/flywire_game/motion.js';

test('slow responses keep moving without crossing the strike area', () => {
  for (const duration of [2.8,3.3,8]) {
    let previous=-19;
    for(let t=0;t<=duration*6;t+=.01){
      const {z,velocity}=approachMotion(t,duration);
      assert.ok(z>=previous && z < -3 && velocity>0);
      previous=z;
    }
    const left=approachMotion(duration-1e-6,duration),right=approachMotion(duration+1e-6,duration);
    assert.ok(Math.abs(left.z-right.z)<1e-4);
    assert.ok(Math.abs(left.velocity-right.velocity)<1e-4);
  }
});
test('early, normal and late results land without a position or velocity jump', () => {
  for(const elapsed of [.1,2.8,4,12]) for(const end of [1.3,6]){
    const {z,velocity}=approachMotion(elapsed,2.8),duration=1.25;
    assert.equal(landingPosition(0,z,end,velocity,duration),z);
    assert.equal(landingPosition(1,z,end,velocity,duration),end);
    const measured=(landingPosition(1e-6,z,end,velocity,duration)-z)/(1e-6*duration);
    assert.ok(Math.abs(measured-velocity)<1e-4);
    let previous=z;
    for(let t=.001;t<1;t+=.001){const position=landingPosition(t,z,end,velocity,duration);assert.ok(position>=previous&&position<=end);previous=position;}
  }
});
test('descent is readable for fast inference and bounded for slow inference', () => {
  assert.equal(descentDuration(.1),2.8);
  assert.equal(descentDuration(30),8);
  assert.ok(descentDuration(4)>descentDuration(2));
});
