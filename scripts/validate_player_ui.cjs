const { chromium } = require(require.resolve('playwright', { paths: [process.env.PLAYER_UI_TOOLS_DIR || process.cwd()] }));
const fs = require('node:fs');
const assert = require('node:assert/strict');
const path = require('node:path');
const baseUrl = process.env.PLAYER_UI_URL || 'http://127.0.0.1:8000';
const pairFixture = process.env.PLAYER_UI_PAIR_FIXTURE ? JSON.parse(fs.readFileSync(process.env.PLAYER_UI_PAIR_FIXTURE,'utf8')) : [{pankti_id:'1',punjabi:'ਪਹਿਲੀ ਪੰਕਤੀ',legacy_gurmukhi:'ਪਹਿਲੀ ਪੰਕਤੀ',english:'First line.'},{pankti_id:'2',punjabi:'ਅਗਲੀ ਪੰਕਤੀ',legacy_gurmukhi:'ਅਗਲੀ ਪੰਕਤੀ',english:'Following line.'}];
const output = process.env.PLAYER_UI_OUTPUT || '/tmp/player-ui-validation';
const raagiArtworkAvailable = fs.existsSync(path.join(__dirname, '../public/darbar-sahib-raagi.png'));
const raagiArtworkUrl = '/static/darbar-sahib-raagi.png';
const raagiName = 'ਭਾਈ ਸੁਖਜੀਤ ਸਿੰਘ ਬ';
fs.mkdirSync(output, {recursive:true});
(async () => {
 const browser = await chromium.launch({executablePath:process.env.CHROMIUM_PATH || '/usr/bin/chromium',headless:true,args:['--no-sandbox']});
 const results=[];
 for (const viewport of [{width:1280,height:800},{width:390,height:844},{width:320,height:700}]) {
  const page=await browser.newPage({viewport,timezoneId:'Australia/Brisbane'});const errors=[];page.on('pageerror',e=>errors.push(e.message));
  let listeners=1234;let revision=0;let waiting=false;let nextLine=false;
  let waitingReason='greeting';let raagi=null;let artworkUrl=raagiArtworkUrl;
  let scheduled=false;let freezePolls=false;
  if (!raagiArtworkAvailable) {
   // Aspect-ratio scaffold tests the overlay only. It does not validate the supplied artwork.
   await page.route('**/static/darbar-sahib-raagi.png', route=>route.fulfill({contentType:'image/svg+xml',body:'<svg xmlns="http://www.w3.org/2000/svg" width="1983" height="793"><rect width="1983" height="793" fill="#fff3d9"/><text x="740" y="250" font-size="35">Layout test fixture</text></svg>'}));
  }
  await page.route('**/static/test-missing-raagi.png',route=>route.fulfill({status:404,body:''}));
  await page.clock.setFixedTime(new Date("2026-10-13T06:00:00Z"));
  await page.route('**/api/live/state?**',route=> {
   if (freezePolls) return route.fulfill({status:204});
   const line=scheduled?pairFixture[0]:nextLine?pairFixture[1]:{punjabi:'ਨਾਮ ਜਪੋ',legacy_gurmukhi:'ਨਾਮ ਜਪੋ',english:'Remember the Name.'};
   return route.fulfill({json:{connected:true,playback_ready:true,audio_url:'/api/live/audio?session_id=fixture',audio_generation:1,playback_delay_seconds:10,listener_count:listeners,waiting_for_shabad:waiting,waiting_reason:waitingReason,raagi,raagi_artwork_url:artworkUrl,
     ...(scheduled?{audio_start_timestamp:1000,pankti_schedule:[
      {audio_timestamp:1000,line:pairFixture[0]},
      {audio_timestamp:1000.25,line:pairFixture[1]},
      {audio_timestamp:1000.7,line:{pankti_id:null,punjabi:'',legacy_gurmukhi:'',english:'',waiting_for_shabad:true,waiting_reason:'greeting',raagi:null}},
      {audio_timestamp:1001.2,line:{pankti_id:null,punjabi:'',legacy_gurmukhi:'',english:'',waiting_for_shabad:true,waiting_reason:'no-shabad',raagi:raagiName}}
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
  await page.waitForTimeout(500);
  assert(!(await page.locator('#raagiArtwork').isVisible()),'Stalled audio must not reach the future raagi card');
  await page.evaluate(()=>{window.mockAudioTime=1.25});
  await page.waitForFunction(()=>!document.getElementById('raagiArtwork').hidden,{},{timeout:800});
  assert.equal(await page.locator('#currentRaagiName').innerText(),raagiName);
  await page.waitForFunction(()=>document.getElementById('raagiImage').naturalWidth===1983);
  assert(!(await page.locator('#defaultWelcomeImage').isVisible()));
  const nameBox=await page.locator('#currentRaagiName').boundingBox();
  const imageBox=await page.locator('#raagiImage').boundingBox();
  assert(nameBox.x>=imageBox.x && nameBox.x+nameBox.width<=imageBox.x+imageBox.width);
  assert(await page.locator('#currentRaagiName').evaluate(n=>n.scrollHeight<=n.clientHeight && n.scrollWidth<=n.clientWidth),'Raagi name fits its panel');
  const raagiDate=await page.locator('.darbar-date-time').boundingBox();
  const raagiCount=await page.locator('.listener-badge').boundingBox();
  assert(raagiDate.x+raagiDate.width<=raagiCount.x || raagiDate.y+raagiDate.height<=raagiCount.y,'Raagi date and listener badge must not overlap');
  assert(raagiCount.x+raagiCount.width<=viewport.width);
  const metadataBox=await page.locator('.artwork-meta').boundingBox();
  assert(Math.abs(raagiCount.x+raagiCount.width-metadataBox.x-metadataBox.width)<2,'Listeners remain right-aligned when metadata wraps');
  await page.screenshot({path:path.join(output,`${raagiArtworkAvailable?'raagi':'raagi-layout-fixture'}-${viewport.width}.png`)});
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
  assert(await page.locator('#defaultWelcomeImage').isVisible());
  await page.screenshot({path:path.join(output,`greeting-${viewport.width}.png`)});
  for (const reason of ['simran','ardas']) {
   waitingReason=reason;
   await page.waitForTimeout(1100);
   assert(await page.locator('#defaultWelcomeImage').isVisible());
   assert(!(await page.locator('#raagiArtwork').isVisible()));
  }
  waitingReason='no-shabad';raagi=raagiName;
  await page.waitForFunction(()=>!document.getElementById('raagiArtwork').hidden);
  // File/date/time gaps all arrive as a null raagi and use the greeting layout.
  raagi=null;
  await page.waitForFunction(()=>document.getElementById('raagiArtwork').hidden);
  assert(await page.locator('#defaultWelcomeImage').isVisible());
  raagi=raagiName;artworkUrl='/static/test-missing-raagi.png';
  await page.waitForFunction(()=>document.getElementById('raagiImage').getAttribute('src')==='/static/test-missing-raagi.png');
  await page.waitForFunction(()=>document.getElementById('raagiArtwork').hidden);
  assert(await page.locator('#defaultWelcomeImage').isVisible());
  artworkUrl=raagiArtworkUrl;
  waiting=false;
  await page.waitForFunction(()=>document.getElementById('welcomeScreen').hidden);
  assert(await page.locator('#panktiGroup').isVisible());
  // A fresh visit returns to the welcome screen; counts remain live.
  waiting=true;waitingReason='no-shabad';raagi=raagiName;
  await page.reload();await page.waitForFunction(()=>!document.getElementById('playToggle').disabled);
  listeners=1;await page.waitForFunction(()=>document.getElementById('listenerCount').textContent==='1');
  assert.equal(await page.locator('#listenerLabel').innerText(),'listener');
  assert(await page.locator('#welcomeScreen').isVisible());assert(!(await page.locator('#panktiGroup').isVisible()));
  assert(await page.locator('#defaultWelcomeImage').isVisible());assert(!(await page.locator('#raagiArtwork').isVisible()));
  const badge=await page.locator('.listener-badge').boundingBox();assert(badge.x>=0 && badge.x+badge.width<=viewport.width);
  assert.deepEqual(errors,[]);results.push({viewport,passed:true});await page.close();
 }
 await browser.close();fs.writeFileSync(path.join(output,'browser-results.json'),JSON.stringify({completed:true,originalArtworkVerified:true,raagiArtworkVerified:raagiArtworkAvailable,raagiOverlayScaffold:!raagiArtworkAvailable,singlePankti:true,firstWordScheduleWithoutPoll:true,stalledAudioHoldsLine:true,timedGreeting:true,timedRaagi:true,missingDutyFallback:true,failedImageFallback:true,results},null,2));console.log(results);
})().catch(e=>{console.error(e);process.exit(1)});
