const { chromium } = require(require.resolve('playwright', { paths: [process.env.PLAYER_UI_TOOLS_DIR || process.cwd()] }));
const fs = require('node:fs');
const assert = require('node:assert/strict');
const path = require('node:path');
const baseUrl = process.env.PLAYER_UI_URL || 'http://127.0.0.1:8000';
const output = process.env.PLAYER_UI_OUTPUT || '/tmp/player-ui-validation';
fs.mkdirSync(output, {recursive:true});
(async () => {
 const browser = await chromium.launch({executablePath:process.env.CHROMIUM_PATH || '/usr/bin/chromium',headless:true,args:['--no-sandbox']});
 const results=[];
 for (const viewport of [{width:1280,height:800},{width:390,height:844}]) {
  const page=await browser.newPage({viewport,timezoneId:'Australia/Brisbane'});const errors=[];page.on('pageerror',e=>errors.push(e.message));
  let listeners=1234;let revision=0;
  await page.route('**/api/live/state?**',route=>route.fulfill({json:{connected:true,playback_ready:true,audio_url:'/api/live/audio?session_id=fixture',audio_generation:1,playback_delay_seconds:10,listener_count:listeners,revision:String(++revision),punjabi:'ਨਾਮ ਜਪੋ',legacy_gurmukhi:'ਨਾਮ ਜਪੋ',english:'Remember the Name.',playback_text:'ਨਾਮ ਜਪੋ'}}));
  await page.addInitScript(() => {
   window.playCalls=0;window.mockPaused=true;
   Object.defineProperty(HTMLMediaElement.prototype,'paused',{get(){return window.mockPaused}});
   HTMLMediaElement.prototype.play=function(){window.playCalls++;return new Promise(resolve=>{window.resolvePlay=resolve})};
   HTMLMediaElement.prototype.load=function(){};
   HTMLMediaElement.prototype.pause=function(){window.mockPaused=true;this.dispatchEvent(new Event('pause'))};
   window.startAudio=()=>{window.mockPaused=false;document.getElementById('liveAudio').dispatchEvent(new Event('playing'));window.resolvePlay()};
  });
  await page.goto(baseUrl.replace(/\/$/, '')+'/?debug');
  await page.waitForFunction(()=>!document.getElementById('playToggle').disabled);
  assert(await page.locator('#welcomeScreen').isVisible());
  assert(await page.locator('.welcome-image').evaluate(img=>img.complete && img.naturalWidth===2172));
  assert(!(await page.locator('#panktiGroup').isVisible()));assert(!(await page.locator('#emptyText').isVisible()));assert(!(await page.locator('#debugPanel').isVisible()));
  assert.equal(await page.locator('#listenerCount').innerText(),'1,234');
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
  // A fresh visit returns to the welcome screen; counts remain live.
  await page.reload();await page.waitForFunction(()=>!document.getElementById('playToggle').disabled);
  listeners=1;await page.waitForFunction(()=>document.getElementById('listenerCount').textContent==='1');
  assert.equal(await page.locator('#listenerLabel').innerText(),'listener');
  assert(await page.locator('#welcomeScreen').isVisible());assert(!(await page.locator('#panktiGroup').isVisible()));
  const badge=await page.locator('.listener-badge').boundingBox();assert(badge.x>=0 && badge.x+badge.width<=viewport.width);
  assert.deepEqual(errors,[]);results.push({viewport,passed:true});await page.close();
 }
 await browser.close();fs.writeFileSync(path.join(output,'browser-results.json'),JSON.stringify({completed:true,artworkVerified:true,results},null,2));console.log(results);
})().catch(e=>{console.error(e);process.exit(1)});
