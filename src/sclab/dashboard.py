"""Self-contained live dashboard served at /dashboard.

Everything (CSS + JS) is inlined so it works from a plain browser or embedded in
a desktop webview with no external assets. It polls /dashboard/stats and drives
a pipeline animation + real-time metrics, and includes a playground that streams
against /v1/chat/completions so the dashboard demonstrates itself.
"""

DASHBOARD_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Matryoshka Inference — Live</title>
<style>
  :root{
    --bg:#0a0e14; --panel:#121821; --panel2:#0d131b; --bd:#1e2733;
    --tx:#e6edf3; --mut:#7d8896; --teal:#2dd4bf; --amber:#f5a623;
    --violet:#a78bfa; --green:#3fb950; --red:#f85149;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--tx);
    font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
  .mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
  a{color:var(--teal)}
  header{display:flex;align-items:center;gap:14px;padding:14px 20px;
    border-bottom:1px solid var(--bd);background:var(--panel2);position:sticky;top:0;z-index:5}
  header h1{font-size:16px;margin:0;font-weight:650;letter-spacing:.2px}
  header .sub{color:var(--mut);font-size:12px}
  .badge{font-size:11px;padding:3px 9px;border-radius:999px;border:1px solid var(--bd);
    color:var(--mut);background:var(--panel)}
  .dot{width:8px;height:8px;border-radius:50%;background:var(--mut);display:inline-block;margin-right:6px}
  .dot.on{background:var(--green);box-shadow:0 0 8px var(--green)}
  .spacer{flex:1}
  main{max-width:1180px;margin:0 auto;padding:20px;display:flex;flex-direction:column;gap:18px}
  .cards{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}
  .card{background:var(--panel);border:1px solid var(--bd);border-radius:12px;padding:14px 16px}
  .card .lbl{color:var(--mut);font-size:11px;text-transform:uppercase;letter-spacing:.6px}
  .card .val{font-size:30px;font-weight:680;margin-top:4px}
  .card .val small{font-size:14px;color:var(--mut);font-weight:500}
  .card.teal .val{color:var(--teal)} .card.amber .val{color:var(--amber)}
  .card.violet .val{color:var(--violet)} .card.green .val{color:var(--green)}
  .panel{background:var(--panel);border:1px solid var(--bd);border-radius:12px;padding:16px}
  .panel h2{font-size:12px;margin:0 0 12px;color:var(--mut);text-transform:uppercase;letter-spacing:.6px}
  /* pipeline */
  .pipe{display:flex;align-items:center;gap:0;flex-wrap:nowrap;overflow-x:auto}
  .node{flex:0 0 auto;min-width:96px;text-align:center;border:1px solid var(--bd);border-radius:10px;
    padding:12px 14px;background:var(--panel2);transition:all .25s;position:relative}
  .node .t{font-weight:600;font-size:13px}
  .node .d{font-size:10.5px;color:var(--mut);margin-top:3px}
  .node.active{border-color:var(--teal);box-shadow:0 0 0 1px var(--teal),0 0 22px rgba(45,212,191,.28);
    background:rgba(45,212,191,.07)}
  .node.active.ar{border-color:var(--amber);box-shadow:0 0 0 1px var(--amber),0 0 22px rgba(245,166,35,.25);
    background:rgba(245,166,35,.07)}
  .arrow{flex:0 0 auto;width:30px;height:2px;background:var(--bd);position:relative}
  .arrow.flow{background:linear-gradient(90deg,var(--teal),transparent);
    animation:flow 1s linear infinite}
  .arrow.flow.ar{background:linear-gradient(90deg,var(--amber),transparent)}
  @keyframes flow{0%{opacity:.3}50%{opacity:1}100%{opacity:.3}}
  .lanes{flex:0 0 auto;display:flex;flex-direction:column;gap:8px}
  .lane{display:flex;align-items:center;gap:8px;opacity:.32;transition:opacity .25s}
  .lane.on{opacity:1}
  .cycle{font-size:16px;color:var(--teal);animation:spin 1.4s linear infinite;display:inline-block}
  @keyframes spin{to{transform:rotate(360deg)}}
  .cap{color:var(--mut);font-size:12px;margin-top:12px}
  .cap b{color:var(--tx)}
  /* chart */
  canvas{width:100%;height:120px;display:block}
  /* sources */
  .bar{height:26px;border-radius:6px;overflow:hidden;display:flex;background:var(--panel2);border:1px solid var(--bd)}
  .bar span{display:block;height:100%}
  .seg-diff{background:var(--teal)} .seg-copy{background:var(--violet)} .seg-ar{background:var(--amber)}
  .legend{display:flex;gap:16px;margin-top:10px;font-size:12px;color:var(--mut);flex-wrap:wrap}
  .sw{width:10px;height:10px;border-radius:3px;display:inline-block;margin-right:6px;vertical-align:middle}
  /* feed */
  table{width:100%;border-collapse:collapse;font-size:12.5px}
  th{text-align:left;color:var(--mut);font-weight:500;padding:6px 8px;border-bottom:1px solid var(--bd);font-size:11px}
  td{padding:6px 8px;border-bottom:1px solid var(--panel2)}
  .chip{font-size:10.5px;padding:2px 8px;border-radius:999px;font-weight:600}
  .chip.diffusion{background:rgba(45,212,191,.14);color:var(--teal)}
  .chip.ar{background:rgba(245,166,35,.14);color:var(--amber)}
  .prev{color:var(--mut);max-width:340px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
  /* playground */
  .pg{display:flex;flex-direction:column;gap:10px}
  .pg .row{display:flex;gap:8px;flex-wrap:wrap}
  textarea{width:100%;min-height:64px;background:var(--panel2);color:var(--tx);border:1px solid var(--bd);
    border-radius:8px;padding:10px;font:13px/1.5 inherit;resize:vertical}
  button{background:var(--teal);color:#06231f;border:0;border-radius:8px;padding:9px 16px;
    font-weight:650;cursor:pointer;font-size:13px}
  button.ghost{background:var(--panel2);color:var(--mut);border:1px solid var(--bd);font-weight:500}
  button:disabled{opacity:.5;cursor:default}
  .out{background:var(--panel2);border:1px solid var(--bd);border-radius:8px;padding:12px;
    min-height:52px;white-space:pre-wrap;font-family:ui-monospace,Menlo,monospace;font-size:12.5px}
  .grid2{display:grid;grid-template-columns:1fr 1fr;gap:18px}
  @media(max-width:820px){.cards{grid-template-columns:repeat(2,1fr)}.grid2{grid-template-columns:1fr}}
  .foot{color:var(--mut);font-size:11.5px;text-align:center;padding:6px 0 22px}
</style>
</head>
<body>
<header>
  <span><span id="dot" class="dot"></span></span>
  <h1>Matryoshka Inference</h1>
  <span class="sub">lossless local acceleration — live</span>
  <span class="spacer"></span>
  <span class="badge" id="b-model">model —</span>
  <span class="badge" id="b-mode">mode —</span>
</header>
<main>

  <div class="cards">
    <div class="card teal"><div class="lbl">Tokens / sec</div><div class="val"><span id="m-tps">0</span><small> tok/s</small></div></div>
    <div class="card violet"><div class="lbl">Accepted / verify pass</div><div class="val"><span id="m-app">1.0</span><small>× vs 1.0 AR</small></div></div>
    <div class="card green"><div class="lbl">Draft acceptance</div><div class="val"><span id="m-acc">0</span><small>%</small></div></div>
    <div class="card amber"><div class="lbl">Speedup vs AR</div><div class="val"><span id="m-spd">—</span><small>×</small></div></div>
  </div>

  <div class="panel">
    <h2>How it works — live pipeline</h2>
    <div class="pipe">
      <div class="node" id="n-prompt"><div class="t">Prompt</div><div class="d" id="n-prompt-d">idle</div></div>
      <div class="arrow" id="a1"></div>
      <div class="node" id="n-route"><div class="t">Router</div><div class="d" id="n-route-d">auto</div></div>
      <div class="arrow" id="a2"></div>
      <div class="lanes">
        <div class="lane" id="lane-diff">
          <div class="node" id="n-draft"><div class="t">Draft <span class="cycle" id="cyc">⟳</span></div><div class="d">diffusion block</div></div>
          <div class="arrow" style="width:18px"></div>
          <div class="node" id="n-verify"><div class="t">Verify</div><div class="d">exact AR pass</div></div>
        </div>
        <div class="lane" id="lane-ar">
          <div class="node ar" id="n-decode"><div class="t">Decode</div><div class="d">plain AR</div></div>
        </div>
      </div>
      <div class="arrow" id="a3"></div>
      <div class="node" id="n-stream"><div class="t">Stream</div><div class="d">to client</div></div>
    </div>
    <div class="cap" id="cap">
      Idle. Structured / reasoning prompts route to <b style="color:var(--teal)">diffusion</b>
      (draft a block, verify it with the exact model — many tokens per pass);
      free-form prose routes to <b style="color:var(--amber)">AR</b> (one verified token per pass).
      A scheduler drops to the AR lane if drafts stop landing, so it's never slower than plain decode by much.
    </div>
  </div>

  <div class="panel">
    <h2>Throughput (tok/s, rolling)</h2>
    <canvas id="chart" width="1120" height="120"></canvas>
  </div>

  <div class="grid2">
    <div class="panel">
      <h2>Token sources — where each token came from</h2>
      <div class="bar" id="srcbar">
        <span class="seg-diff" id="s-diff" style="width:0"></span>
        <span class="seg-copy" id="s-copy" style="width:0"></span>
        <span class="seg-ar" id="s-ar" style="width:0"></span>
      </div>
      <div class="legend">
        <span><span class="sw seg-diff"></span>diffusion (verified draft) <b id="l-diff" class="mono">0</b></span>
        <span><span class="sw seg-copy"></span>copy (repeat) <b id="l-copy" class="mono">0</b></span>
        <span><span class="sw seg-ar"></span>AR correction <b id="l-ar" class="mono">0</b></span>
      </div>
    </div>
    <div class="panel">
      <h2>Totals</h2>
      <table>
        <tr><td>Requests served</td><td class="mono" id="t-req" style="text-align:right">0</td></tr>
        <tr><td>Tokens generated</td><td class="mono" id="t-tok" style="text-align:right">0</td></tr>
        <tr><td>Routed to diffusion / AR</td><td class="mono" id="t-mode" style="text-align:right">0 / 0</td></tr>
        <tr><td>Draft positions pruned</td><td class="mono" id="t-prune" style="text-align:right">0</td></tr>
      </table>
    </div>
  </div>

  <div class="panel">
    <h2>Recent requests</h2>
    <table>
      <thead><tr><th>mode</th><th>tokens</th><th>tok/s</th><th>acc/pass</th><th>accept</th><th>ms</th><th>prompt</th></tr></thead>
      <tbody id="feed"><tr><td colspan="7" style="color:var(--mut)">no requests yet — try the playground below, or point Hermes at this server</td></tr></tbody>
    </table>
  </div>

  <div class="panel pg">
    <h2>Playground — watch it work</h2>
    <div class="row">
      <button class="ghost" data-p="Output a JSON array of 6 product objects with fields sku, name, price, stock.">JSON</button>
      <button class="ghost" data-p="Solve step by step: a train leaves at 3pm at 60mph, another at 4pm at 80mph same direction — when does it catch up?">Reasoning</button>
      <button class="ghost" data-p="Write a Python function that merges two sorted lists.">Code</button>
      <button class="ghost" data-p="Describe a quiet morning by the sea in a short paragraph.">Prose</button>
    </div>
    <textarea id="pg-in" placeholder="Type a prompt and press Send…">Output a JSON array of 6 product objects with fields sku, name, price, stock.</textarea>
    <div class="row"><button id="pg-send">Send</button><span class="sub" id="pg-note" style="color:var(--mut);align-self:center"></span></div>
    <div class="out" id="pg-out"></div>
  </div>

  <div class="foot">Matryoshka Inference · dashboard polls <span class="mono">/dashboard/stats</span> · OpenAI API at <span class="mono">/v1</span></div>
</main>

<script>
const $=id=>document.getElementById(id);
const fmt=n=>Number(n).toLocaleString();

function setPipe(live){
  const active=live.active, mode=live.mode, ph=live.phase;
  const diffOn = active && mode==='diffusion';
  const arOn = active && mode==='ar';
  $('lane-diff').classList.toggle('on', diffOn || !active);
  $('lane-ar').classList.toggle('on', arOn || !active);
  // node states
  $('n-prompt').classList.toggle('active', active);
  $('n-route').classList.toggle('active', active);
  $('n-stream').classList.toggle('active', active);
  $('n-draft').classList.toggle('active', diffOn);
  $('n-verify').classList.toggle('active', diffOn);
  $('n-decode').classList.toggle('active', arOn);
  $('n-decode').classList.toggle('ar', true);
  // arrows
  ['a1','a2','a3'].forEach(a=>{const e=$(a);e.classList.toggle('flow',active);e.classList.toggle('ar',arOn)});
  $('n-prompt-d').textContent = active ? (live.tokens+' tok') : 'idle';
  $('n-route-d').textContent = active ? mode.toUpperCase() : 'auto';
  $('cyc').style.display = diffOn ? 'inline-block':'none';
  if(active){
    const src = diffOn ? 'var(--teal)' : 'var(--amber)';
    $('cap').innerHTML = `Generating in <b style="color:${src}">${mode.toUpperCase()}</b> — `
      + (diffOn
         ? `drafting a block and verifying it with the exact model: <b>${live.accepted_per_pass}</b> accepted tokens per pass, <b>${Math.round(live.acceptance_rate*100)}%</b> draft acceptance.`
         : `one verified token per pass (prose drafts poorly, so the router picked plain AR).`);
  }
}

let chart=$('chart'), cx=chart.getContext('2d');
function drawChart(spark){
  const w=chart.width,h=chart.height; cx.clearRect(0,0,w,h);
  if(!spark.length) return;
  const mx=Math.max(...spark,1), pad=6;
  cx.strokeStyle='#1e2733';cx.lineWidth=1;
  for(let i=0;i<=4;i++){const y=pad+(h-2*pad)*i/4;cx.beginPath();cx.moveTo(0,y);cx.lineTo(w,y);cx.stroke();}
  cx.beginPath();
  spark.forEach((v,i)=>{const x=w*i/(spark.length-1||1);const y=h-pad-(h-2*pad)*(v/mx);i?cx.lineTo(x,y):cx.moveTo(x,y);});
  cx.strokeStyle='#2dd4bf';cx.lineWidth=2;cx.stroke();
  cx.lineTo(w,h);cx.lineTo(0,h);cx.closePath();
  cx.fillStyle='rgba(45,212,191,.10)';cx.fill();
  cx.fillStyle='#7d8896';cx.font='11px monospace';cx.fillText(mx.toFixed(0)+' tok/s',6,14);
}

function render(s){
  $('dot').classList.add('on');
  $('b-model').textContent = 'model · '+(s.server?.model||'—');
  $('b-mode').textContent = 'mode · '+(s.server?.mode||'—');
  const lv=s.live, r=s.rates;
  $('m-tps').textContent = lv.active ? lv.tok_s : (r.diffusion_tok_s||0);
  $('m-app').textContent = (lv.active?lv.accepted_per_pass:(s.history[0]?.accepted_per_pass))||'1.0';
  $('m-acc').textContent = Math.round(((lv.active?lv.acceptance_rate:(s.history[0]?.acceptance_rate))||0)*100);
  $('m-spd').textContent = r.speedup_vs_ar ? r.speedup_vs_ar : '—';
  setPipe(lv);
  drawChart(s.spark);
  // sources
  const src=s.totals.tokens_by_source, tot=(src.diffusion+src.copy+src.ar)||1;
  $('s-diff').style.width=(100*src.diffusion/tot)+'%';
  $('s-copy').style.width=(100*src.copy/tot)+'%';
  $('s-ar').style.width=(100*src.ar/tot)+'%';
  $('l-diff').textContent=fmt(src.diffusion);$('l-copy').textContent=fmt(src.copy);$('l-ar').textContent=fmt(src.ar);
  // totals
  $('t-req').textContent=fmt(s.totals.requests);
  $('t-tok').textContent=fmt(s.totals.tokens);
  $('t-mode').textContent=s.totals.by_mode.diffusion+' / '+s.totals.by_mode.ar;
  $('t-prune').textContent=fmt(s.totals.pruned_positions);
  // feed
  const fb=$('feed');
  if(s.history.length){
    fb.innerHTML=s.history.map(h=>`<tr>
      <td><span class="chip ${h.mode}">${h.mode}</span></td>
      <td class="mono">${h.tokens}</td><td class="mono">${h.tok_s}</td>
      <td class="mono">${h.accepted_per_pass}</td><td class="mono">${Math.round(h.acceptance_rate*100)}%</td>
      <td class="mono">${h.ms}</td><td class="prev">${(h.prompt_preview||'').replace(/</g,'&lt;')}</td></tr>`).join('');
  }
}

async function poll(){
  try{const r=await fetch('/dashboard/stats');render(await r.json());}
  catch(e){$('dot').classList.remove('on');}
}
setInterval(poll,350); poll();

// playground
document.querySelectorAll('[data-p]').forEach(b=>b.onclick=()=>{$('pg-in').value=b.dataset.p;});
$('pg-send').onclick=async()=>{
  const btn=$('pg-send'), out=$('pg-out'); btn.disabled=true; out.textContent='';
  $('pg-note').textContent='streaming…';
  try{
    const res=await fetch('/v1/chat/completions',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({model:'local',stream:true,max_tokens:400,
        messages:[{role:'user',content:$('pg-in').value}]})});
    const rd=res.body.getReader(), dec=new TextDecoder(); let buf='';
    while(true){const {done,value}=await rd.read(); if(done)break;
      buf+=dec.decode(value,{stream:true}); let i;
      while((i=buf.indexOf('\n\n'))>=0){const line=buf.slice(0,i);buf=buf.slice(i+2);
        if(line.startsWith('data: ')){const d=line.slice(6); if(d==='[DONE]')continue;
          try{const j=JSON.parse(d);const c=j.choices?.[0]?.delta?.content;if(c)out.textContent+=c;}catch(e){}}}}
    $('pg-note').textContent='done';
  }catch(e){$('pg-note').textContent='error: '+e.message;}
  finally{btn.disabled=false;}
};
</script>
</body>
</html>"""
