const { chromium } = require(require.resolve('playwright', { paths: [process.env.PLAYER_UI_TOOLS_DIR || process.cwd()] }));
const fs = require('node:fs');
const assert = require('node:assert/strict');
const path = require('node:path');
const baseUrl = process.env.PLAYER_UI_URL || 'http://127.0.0.1:8000';
const pairFixture = process.env.PLAYER_UI_PAIR_FIXTURE ? JSON.parse(fs.readFileSync(process.env.PLAYER_UI_PAIR_FIXTURE,'utf8')) : [{pankti_id:'1',punjabi:'ਪਹਿਲੀ ਪੰਕਤੀ',legacy_gurmukhi:'ਪਹਿਲੀ ਪੰਕਤੀ',english:'First line.'},{pankti_id:'2',punjabi:'ਅਗਲੀ ਪੰਕਤੀ',legacy_gurmukhi:'ਅਗਲੀ ਪੰਕਤੀ',english:'Following line.'}];
const output = process.env.PLAYER_UI_OUTPUT || '/tmp/player-ui-validation';
fs.mkdirSync(output, {recursive:true});
(async () => {
 const browser = await chromium.launch({executablePath:process.env.CHROMIUM_PATH || '/usr/bin/chromium',headless:true,args:['--no-sandbox']});
 const results=[];
 for (const viewport of [{width:1280,height:800},{width:390,height:844},{width:320,height:700}]) {
  const page=await browser.newPage({viewport,timezoneId:'Australia/Brisbane'});const errors=[];page.on('pageerror',e=>errors.push(e.message));
  let listeners=1234;let revision=0;let waiting=false;let nextLine=false;let waitingReason="greeting";
  let scheduled=false;let freezePolls=false;
  await page.clock.setFixedTime(new Date("2026-10-13T06:00:00Z"));
  await page.route('**/api/live/state?**',route=> {
   if (freezePolls) return route.fulfill({status:204});
   const line=scheduled?pairFixture[0]:nextLine?pairFixture[1]:{punjabi:'ਨਾਮ ਜਪੋ',legacy_gurmukhi:'ਨਾਮ ਜਪੋ',english:'Remember the Name.'};
   return route.fulfill({json:{connected:true,playback_ready:true,audio_url:'/api/live/audio?session_id=fixture',audio_generation:1,playback_delay_seconds:10,listener_count:listeners,waiting_for_shabad:waiting,waiting_reason:waitingReason,raagi:"SHOULD NOT DISPLAY",raagi_artwork_url:"/static/absent.png",
     ...(scheduled?{audio_start_timestamp:1000,pankti_schedule:[
      {audio_timestamp:1000,line:pairFixture[0]},
      {audio_timestamp:1000.25,line:pairFixture[1]},
      {audio_timestamp:1000.7,line:{pankti_id:null,punjabi:'',legacy_gurmukhi:'',english:'',waiting_for_shabad:true,waiting_reason:'role-ending'}},
      {audio_timestamp:1001.2,line:pairFixture[0]}
     ]}:{}),
     // A stale paired field must never make the new UI show an older line.
     display_panktis:nextLine?pairFixture:[],revision:String(++revision),...line,playback_text:'ਨਾਮ ਜਪੋ'}});
  });
  await page.addInitScript(() => {
   window.playCalls=0;window.mockPaused=true;window.mockAudioTime=0;
   Object.defineProperty(HTMLMediaElement.prototype,'paused',{get(){return window.mockPaused}});
   Object.defineProperty(HTMLMediaElement.prototype,'currentTime',{get(){return window.mockAudioTime},set(value){window.mockAudioTime=value}});
   HTMLMediaElement.prototype.play=function(){window.playCalls++;return new Promise(resolve=>{window.resolvePlay=resolve})};
   HTMLMediaElement.prototype.load=function(){};
   HTMLMediaElement.prototype.pause=function(){window.mockPaused=true;this.dispatchEvent(new Event('pause'))};
   window.startAudio=()=>{window.mockPaused=false;document.getElementById('liveAudio').dispatchEvent(new Event('playing'));window.resolvePlay()};
  });
  await page.goto(baseUrl.replace(/\/$/, '')+'/?debug');
  await page.waitForFunction(()=>!document.getElementById('playToggle').disabled);
  assert(await page.locator('#welcomeScreen').isVisible());
  assert(await page.locator('#defaultWelcomeImage').evaluate(img=>img.complete && img.naturalWidth===2172));
  assert(!(await page.locator('#panktiGroup').isVisible()));assert(!(await page.locator('#emptyText').isVisible()));assert(!(await page.locator('#debugPanel').isVisible()));
  assert.equal(await page.locator('#listenerCount').innerText(),'1,234');
  assert.equal(await page.locator('#darbarNanakshahiDate').innerText(),'੨੭ ਅੱਸੂ ੫੫੮');
  assert.equal(await page.locator('#darbarIndiaTime').innerText(),'ਅੰਮ੍ਰਿਤਸਰ 11:30 am');
  assert.equal(await page.locator('#darbarIndiaTime').evaluate(n=>getComputedStyle(n).fontSize),'18px');
  assert.equal(await page.locator('#darbarNanakshahiDate').evaluate(n=>getComputedStyle(n).fontSize),'18px');
  assert.equal(await page.locator('#currentUserTime').innerText(),'Start Listening');
  const dateBox=await page.locator('.darbar-date-time').boundingBox();
  const countBox=await page.locator('.listener-badge').boundingBox();
  assert(dateBox.x+dateBox.width<=countBox.x || dateBox.y+dateBox.height<=countBox.y, 'Date and listeners must not overlap');
  assert(countBox.x+countBox.width<=viewport.width);
  await page.click('#playToggle');
  assert(await page.locator('#playToggle').isDisabled());assert.equal(await page.locator('#playToggle').getAttribute('aria-busy'),'true');
  assert(await page.locator('.play-spinner').isVisible());assert(!(await page.locator('#playIcon').isVisible()));
  assert(await page.locator('#welcomeScreen').isVisible());assert(await page.locator('#startingSoon').isVisible());assert(!(await page.locator('#panktiGroup').isVisible()));
  await page.evaluate(()=>document.getElementById('playToggle').click());assert.equal(await page.evaluate(()=>window.playCalls),1);
  await page.screenshot({path:path.join(output,`loading-${viewport.width}.png`)});
  await page.evaluate(()=>window.startAudio());
  await page.waitForFunction(()=>!document.getElementById('playToggle').disabled);
  assert(!(await page.locator('#welcomeScreen').isVisible()));assert(await page.locator('#panktiGroup').isVisible());assert(await page.locator('#debugPanel').isVisible());
  assert(!(await page.locator('.play-spinner').isVisible()));assert(await page.locator('#stopIcon').isVisible());
  await page.evaluate(()=>{window.mockPaused=true;document.getElementById('liveAudio').dispatchEvent(new Event('pause'))});
  assert(await page.locator('#playToggle').isDisabled());assert(await page.locator('.play-spinner').isVisible());
  assert(!(await page.locator('#welcomeScreen').isVisible()));assert(await page.locator('#panktiGroup').isVisible());
  await page.waitForFunction(()=>window.playCalls===2,{},{timeout:6000});
  await page.evaluate(()=>window.startAudio());
  await page.waitForFunction(()=>!document.getElementById('playToggle').disabled);
  assert(!(await page.locator('#welcomeScreen').isVisible()));assert(await page.locator('#panktiGroup').isVisible());
  nextLine=true;
  await page.waitForFunction(expected=>document.getElementById('english').textContent===expected,pairFixture[1].english);
  assert.equal(await page.locator('#punjabi').innerText(),pairFixture[1].punjabi.replace(/[.,;](?=\s|$)/g,''));
  assert.equal(await page.locator('.pankti-entry').count(),1);
  assert.equal(await page.locator('#secondPankti').count(),0);
  assert(!(await page.locator('#panktiGroup').evaluate(n=>n.classList.contains('is-paired'))));
  await page.screenshot({path:path.join(output,`single-${viewport.width}.png`)});
  // One fetched schedule must follow the media clock without another state body.
  scheduled=true;
  await page.waitForFunction(expected=>document.getElementById('english').textContent===expected,pairFixture[0].english);
  freezePolls=true;
  await page.waitForTimeout(400); // Wall time passes but stalled audio stays at zero.
  assert.equal(await page.locator('#english').innerText(),pairFixture[0].english);
  await page.evaluate(()=>{window.mockAudioTime=.3}); // No timeupdate or new poll.
  await page.waitForFunction(expected=>document.getElementById('english').textContent===expected,pairFixture[1].english,{timeout:800});
  assert.equal(await page.locator('.pankti-entry').count(),1);
  await page.waitForTimeout(450);
  assert(!(await page.locator('#welcomeScreen').isVisible()),'Stalled audio must not reach the future greeting');
  await page.evaluate(()=>{window.mockAudioTime=.75});
  await page.waitForFunction(()=>!document.getElementById('welcomeScreen').hidden,{},{timeout:800});
  assert(!(await page.locator('#panktiGroup').isVisible()));
  assert(await page.locator('#defaultWelcomeImage').isVisible());
  await page.evaluate(()=>{window.mockAudioTime=1.25});
  await page.waitForFunction(()=>document.getElementById('welcomeScreen').hidden,{},{timeout:800});
  assert.equal(await page.locator('#english').innerText(),pairFixture[0].english);
  scheduled=false;freezePolls=false;
  await page.evaluate(()=>{window.mockAudioTime=0});
  nextLine=false;
  await page.waitForFunction(()=>document.getElementById('english').textContent==='Remember the Name.');
  waiting=true;
  await page.waitForFunction(()=>!document.getElementById('welcomeScreen').hidden);
  assert(!(await page.locator('#panktiGroup').isVisible()));
  assert(!(await page.locator('#emptyText').isVisible()));
  assert.equal(await page.locator('#startingSoon').innerText(),'Waiting for next shabad…');
  assert(!(await page.locator('#playToggle').isDisabled()));
  assert(await page.locator('#stopIcon').isVisible());
  assert.notEqual(await page.locator('#currentUserTime').innerText(),'Start Listening');
  await page.screenshot({path:path.join(output,`greeting-${viewport.width}.png`)});
  for (const reason of ['simran','ardas','role-ending','no-shabad']) {
   waitingReason=reason;
   await page.waitForTimeout(1100);
   assert(await page.locator('#defaultWelcomeImage').isVisible());
   assert(!(await page.locator('#panktiGroup').isVisible()));
   assert.equal(await page.locator('#currentRaagiName').count(),0);
   assert.equal(await page.locator('#raagiArtwork').count(),0);
  }
  waiting=false;
  await page.waitForFunction(()=>document.getElementById('welcomeScreen').hidden);
  assert(await page.locator('#panktiGroup').isVisible());
  // A fresh visit returns to the welcome screen; counts remain live.
  await page.reload();await page.waitForFunction(()=>!document.getElementById('playToggle').disabled);
  listeners=1;await page.waitForFunction(()=>document.getElementById('listenerCount').textContent==='1');
  assert.equal(await page.locator('#listenerLabel').innerText(),'listener');
  assert(await page.locator('#welcomeScreen').isVisible());assert(!(await page.locator('#panktiGroup').isVisible()));
  const badge=await page.locator('.listener-badge').boundingBox();assert(badge.x>=0 && badge.x+badge.width<=viewport.width);
  assert.deepEqual(errors,[]);results.push({viewport,passed:true});await page.close();
 }
 await browser.close();fs.writeFileSync(path.join(output,'browser-results.json'),JSON.stringify({completed:true,artworkVerified:true,singlePankti:true,firstWordScheduleWithoutPoll:true,stalledAudioHoldsLine:true,timedWaitingAndResume:true,noRaagiName:true,allWaitingReasons:true,results},null,2));console.log(results);
})().catch(e=>{console.error(e);process.exit(1)});
