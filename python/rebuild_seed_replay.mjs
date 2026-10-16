// Rebuild only provably history-free gate calls; the ramp did not persist generated dialogue.
import {readFileSync,writeFileSync,mkdirSync} from 'node:fs';
import {resolve,dirname} from 'node:path';
import {pathToFileURL} from 'node:url';
import {createHash} from 'node:crypto';
const arg=(name)=>{const i=process.argv.indexOf('--'+name);if(i<0)throw Error('Missing --'+name);return process.argv[i+1];};
const repo=resolve(arg('server')),artifact=resolve(arg('artifact')),output=resolve(arg('output'));
const {participationContext}=await import(pathToFileURL(resolve(repo,'server/participation-gate.mjs')));
const {buildSchedule}=await import(pathToFileURL(resolve(repo,'deploy/ai-player-test-support.mjs')));
const source=readFileSync(resolve(repo,'deploy/ai-player-test.mjs'),'utf8');
function literal(name){
 const match=source.match(new RegExp('const '+name+'=(\\[[^;]+\\]);'));
 if(!match)throw Error('Missing harness prompt array '+name);
 return JSON.parse(match[1].replaceAll("'",'"')); // These checked harness arrays contain simple single-quoted strings, never executable expressions.
}
const social=literal('SOCIAL'),game=literal('GAME'),prefaces=literal('PREFACES');
const report=JSON.parse(readFileSync(artifact,'utf8')),stage=report.stages.find(s=>s.players===100);
if(!stage)throw Error('A saved 100-player stage is required');
const catalog=JSON.parse(readFileSync(resolve(repo,'server/followers-data.json'),'utf8')).companions,ids=Object.keys(catalog);
const schedule=buildSchedule({players:100,...report.config}),key=e=>e.player+'/'+e.turn;
const saved=new Map(stage.results.map(row=>[key(row),row]));
const uncertain=new Set(),rows=[];
for(const event of schedule){
 const result=saved.get(key(event));
 if(!result)throw Error('Incomplete saved schedule');
 if(event.prompt!==result.prompt||event.kind!==result.kind||event.nearby!==result.nearby)throw Error('Seed reconstruction differs from saved events');
 if(event.kind!=='companion')continue;
 if(event.nearby&&!uncertain.has(event.player)){
  const candidates=Array.from({length:report.config.nearbyCount},(_,i)=>ids[(event.player+i)%ids.length]);
  const isSocial=event.prompt%5!==0,choices=isSocial?social:game;
  const message=prefaces[Math.floor(event.prompt/101)%prefaces.length]+choices[Math.floor(event.prompt/5)%choices.length];
  for(let i=0;i<candidates.length;i++)rows.push({id:'ramp-100-'+key(event).replace('/','-')+'-'+i,
   task:'participation',label:i===0&&isSocial?'join':null,cohort:i===0?'hired_opening':'bystander_opening',
   family_id:'ramp-player-'+event.player,split:'regression',source_artifact:artifact,
   state:participationContext(catalog[candidates[i]],{message,playerName:'Traveller '+(event.player+1),history:[],conversation:false,hired:i===0,nearbyNames:candidates.map(id=>catalog[id].name)}),
   source_outcome:result.outcome,source_any_join:result.gates?.some(g=>g.join)??null,
   reconstruction:'Exact empty-history input: no earlier companion request was dispatched for this player; seeded prompt and saved event agree.'});
 }
 // Once a generation was attempted, history may exist. Never substitute invented turns for it.
 if(!event.nearby||result.outcome!=='quiet')uncertain.add(event.player);
}
mkdirSync(dirname(output),{recursive:true});
writeFileSync(output,rows.map(row=>JSON.stringify(row)).join('\n')+'\n');
const manifest={source:artifact,source_sha256:createHash('sha256').update(readFileSync(artifact)).digest('hex'),
 harness_sha256:createHash('sha256').update(source).digest('hex'),catalog_sha256:createHash('sha256').update(readFileSync(resolve(repo,'server/followers-data.json'))).digest('hex'),
 rows:rows.length,hired_social:rows.filter(r=>r.label==='join').length,
 limitations:['Only provably empty-history inputs are reconstructed.','Saved per-candidate gate results lack candidate IDs; do not assign source scores to candidates.','Current checked-in catalog is used; original run did not record its hash.','No full-history contexts were fabricated.','Regression only: these messages are excluded from training.']};
writeFileSync(output+'.manifest.json',JSON.stringify(manifest,null,2)+'\n');
console.log(JSON.stringify(manifest,null,2));
