const $ = (q)=>document.querySelector(q);
const state = { map:null, snapshot:null, selected:null, socket:null, metrics:null };

const api = async (url, options={}) => {
  const r = await fetch(url, {headers:{"content-type":"application/json",...(options.headers||{})},...options});
  if (r.status === 401) throw new Error("AUTH");
  if (!r.ok) throw new Error((await r.json().catch(()=>({detail:r.statusText}))).detail || "Erro");
  return r.json();
};
const pct = v => ((v||0)*100).toFixed(1)+"%";
const esc = s => String(s ?? "").replace(/[&<>"']/g, c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#039;"}[c]));

async function boot(){
  try{
    await api("/api/auth/me");
    $("#loginScreen").classList.add("hidden");
    $("#app").classList.remove("hidden");
    const status=await api("/api/lab/status");
    if(!status.configured) $("#setupScreen").classList.remove("hidden");
    else{
      await loadMap();
      connect();
      if(!status.running) await api("/api/lab/start",{method:"POST"});
    }
  }catch(e){ $("#loginScreen").classList.remove("hidden"); }
}

$("#loginForm").addEventListener("submit",async e=>{
  e.preventDefault(); $("#loginError").textContent="";
  try{
    await api("/api/auth/login",{method:"POST",body:JSON.stringify({username:$("#username").value,password:$("#password").value})});
    $("#loginScreen").classList.add("hidden"); $("#app").classList.remove("hidden");
    const s=await api("/api/lab/status");
    if(!s.configured) $("#setupScreen").classList.remove("hidden"); else {await loadMap();connect();}
  }catch(e){$("#loginError").textContent="Login inválido."}
});

$("#setupForm").addEventListener("submit",async e=>{
  e.preventDefault();
  await api("/api/lab/configure",{method:"POST",body:JSON.stringify({
    n_totems:+$("#nTotems").value,radius_km:+$("#radiusKm").value,seed:+$("#seed").value
  })});
  await loadMap(); await api("/api/lab/start",{method:"POST"});
  $("#setupScreen").classList.add("hidden"); connect();
});

async function loadMap(){ state.map=await api("/api/lab/map"); resize(); }

function connect(){
  if(state.socket) try{state.socket.close()}catch(_){}
  const proto=location.protocol==="https:"?"wss":"ws";
  state.socket=new WebSocket(proto+"://"+location.host+"/ws/lab");
  state.socket.onmessage=e=>{state.snapshot=JSON.parse(e.data);updateUI();draw()};
  state.socket.onclose=()=>setTimeout(()=>{if(!document.hidden)connect()},1200);
}

$("#pauseBtn").addEventListener("click",async()=>{
  const paused=!(state.snapshot?.paused);
  await api("/api/lab/pause",{method:"POST",body:JSON.stringify({paused})});
});
document.querySelectorAll("[data-speed]").forEach(b=>b.addEventListener("click",async()=>{
  const speed=+b.dataset.speed;
  await api("/api/lab/speed",{method:"POST",body:JSON.stringify({speed})});
  document.querySelectorAll("[data-speed]").forEach(x=>x.classList.toggle("active",x===b));
}));
$("#creativesBtn").onclick=()=>openDrawer("creativesDrawer",loadCreatives);
$("#metricsBtn").onclick=()=>openDrawer("metricsDrawer",loadMetrics);
document.querySelectorAll("[data-close]").forEach(b=>b.onclick=()=>$("#"+b.dataset.close).classList.remove("open"));
function openDrawer(id,loader){$("#"+id).classList.add("open");loader&&loader()}

$("#newCreativeBtn").onclick=()=>$("#creativeForm").classList.toggle("hidden");
$("#saveCreativeBtn").onclick=async()=>{
  const body={
    ad_id:$("#adId").value.trim(),name:$("#adName").value.trim(),
    category:$("#adCategory").value.trim()||null,
    tags:$("#adTags").value.split(",").map(x=>x.trim()).filter(Boolean),
    duration_seconds:+$("#adDuration").value,daily_budget:+$("#adBudget").value,
    cost_per_play:.06,active:true
  };
  if(!body.ad_id||!body.name)return;
  await api("/api/lab/creatives",{method:"POST",body:JSON.stringify(body)});
  $("#creativeForm").classList.add("hidden");await loadCreatives();
};

async function loadCreatives(){
  const ads=await api("/api/lab/creatives");
  $("#creativeList").innerHTML=ads.map(a=>{
    const p=Math.min(100,(a.spent_today/a.daily_budget)*100);
    return '<div class="creative"><div class="creative-head"><div><h3>'+esc(a.name)+'</h3><small>'+esc(a.ad_id)+' · '+esc(a.category||"sem categoria")+'</small></div><div class="budget">R$ '+a.spent_today.toFixed(2)+' / '+a.daily_budget.toFixed(0)+'</div></div><div class="tag-row">'+a.tags.map(t=>'<span class="tag">'+esc(t)+'</span>').join("")+'</div><div class="progress"><i style="width:'+p+'%"></i></div><small>'+a.plays+' exibições · reward '+pct(a.observed_reward)+' · '+a.observations.toFixed(0)+' evidências</small></div>';
  }).join("");
}

function updateUI(){
  const s=state.snapshot;if(!s?.configured)return;
  const d=new Date(s.sim_time);$("#simClock").textContent=d.toLocaleString("pt-BR",{weekday:"short",day:"2-digit",month:"short",hour:"2-digit",minute:"2-digit"});
  $("#peopleCount").textContent=s.people_count.toLocaleString("pt-BR");
  $("#decisionCount").textContent=s.metrics.decisions.toLocaleString("pt-BR");
  $("#aiReward").textContent=pct(s.metrics.observed_reward);
  $("#randomReward").textContent=pct(s.metrics.random_baseline);
  const up=s.metrics.random_baseline>0?s.metrics.observed_reward/s.metrics.random_baseline-1:0;
  $("#uplift").textContent=(up>=0?"+":"")+pct(up);
  $("#uplift").style.color=up>=0?"#37d39b":"#ff7c88";
  $("#uncertainty").textContent=Number(s.metrics.uncertainty||0).toFixed(4);
  $("#pauseBtn").textContent=s.paused?"Continuar":"Pausar";
  if(state.selected) updateTotem(state.selected);
}

const canvas=$("#cityCanvas"),ctx=canvas.getContext("2d");
function resize(){const r=canvas.getBoundingClientRect();canvas.width=Math.max(1,r.width*devicePixelRatio);canvas.height=Math.max(1,r.height*devicePixelRatio);draw()}
window.addEventListener("resize",resize);
const project=(x,y)=>{
  const rad=state.map?.radius_km||1, pad=35*devicePixelRatio;
  const w=canvas.width-pad*2,h=canvas.height-pad*2;
  const scale=Math.min(w,h)/(rad*2);
  return [canvas.width/2+x*scale,canvas.height/2-y*scale,scale]
};
function draw(){
  if(!state.map||!state.snapshot)return;
  ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.fillStyle="#090a0e";ctx.fillRect(0,0,canvas.width,canvas.height);
  for(const z of state.map.zones){
    const [x,y,sc]=project(z.x,z.y);ctx.beginPath();ctx.arc(x,y,z.radius_km*sc,0,Math.PI*2);
    ctx.fillStyle=z.kind==="downtown"?"rgba(139,92,246,.055)":"rgba(255,255,255,.018)";ctx.fill();
  }
  ctx.lineCap="round";
  for(const r of state.map.roads){
    const [x1,y1]=project(r.x1,r.y1),[x2,y2]=project(r.x2,r.y2);
    ctx.beginPath();ctx.moveTo(x1,y1);ctx.lineTo(x2,y2);
    ctx.strokeStyle=r.kind==="arterial"?"#343844":r.kind==="collector"?"#242832":"#181b22";
    ctx.lineWidth=(r.kind==="arterial"?2.2:r.kind==="collector"?1.4:.8)*devicePixelRatio;ctx.stroke();
  }
  for(const p of state.snapshot.people){
    const [x,y]=project(p.x,p.y);ctx.beginPath();ctx.arc(x,y,1.25*devicePixelRatio,0,Math.PI*2);
    ctx.fillStyle=p.gender==="F"?"#d5c8f3":"#aab6cc";ctx.globalAlpha=.72;ctx.fill();
  }
  ctx.globalAlpha=1;
  for(const t of state.snapshot.totems){
    const [x,y]=project(t.x,t.y);const sel=state.selected===t.id;
    if(sel){ctx.beginPath();ctx.arc(x,y,9*devicePixelRatio,0,Math.PI*2);ctx.strokeStyle="#bda7ff";ctx.lineWidth=2*devicePixelRatio;ctx.stroke()}
    ctx.fillStyle="#8b5cf6";ctx.fillRect(x-3.4*devicePixelRatio,y-3.4*devicePixelRatio,6.8*devicePixelRatio,6.8*devicePixelRatio);
  }
}

canvas.addEventListener("click",e=>{
  if(!state.snapshot)return;
  const rect=canvas.getBoundingClientRect(),mx=(e.clientX-rect.left)*devicePixelRatio,my=(e.clientY-rect.top)*devicePixelRatio;
  let best=null,bd=Infinity;
  for(const t of state.snapshot.totems){const [x,y]=project(t.x,t.y),d=Math.hypot(x-mx,y-my);if(d<bd){bd=d;best=t}}
  if(best&&bd<16*devicePixelRatio){state.selected=best.id;updateTotem(best.id);draw()}
});

async function updateTotem(id){
  try{
    const t=await api("/api/lab/totems/"+encodeURIComponent(id));
    $("#totemTitle").textContent=t.id+" · "+t.region;
    $("#totemSubtitle").textContent=t.current_ad?("Exibindo "+t.current_ad.name):"Aguardando decisão";
    const rows=[
      ["Alcance",t.reach],["Impressões",t.impressions],["Taxa de olhar",pct(t.capture_rate)],
      ["Retenção",t.avg_view_seconds.toFixed(2)+" s"],["Conclusão",pct(t.completion_rate)],
      ["Reward",pct(t.reward)],["Público feminino",pct(t.female_share)],
      ["Idade média",t.mean_age.toFixed(1)],["Fluxo",t.flow_per_minute.toFixed(1)+"/min"],["Exibições",t.plays]
    ];
    $("#totemMetrics").innerHTML=rows.map(r=>'<div class="detail"><span>'+esc(r[0])+'</span><strong>'+esc(r[1])+'</strong></div>').join("");
  }catch(_){}
}

async function loadMetrics(){
  state.metrics=await api("/api/lab/metrics");
  $("#metricAi").textContent=pct(state.metrics.observed_reward);
  $("#metricRandom").textContent=pct(state.metrics.random_baseline);
  $("#metricOracle").textContent=pct(state.metrics.oracle_ceiling);
  $("#metricUplift").textContent=(state.metrics.uplift_vs_random>=0?"+":"")+pct(state.metrics.uplift_vs_random);
  chart($("#rewardChart"),state.metrics.history,[
    ["observed_reward","#9f7aea"],["random_baseline","#77808f"],["oracle_ceiling","#37d39b"]
  ],0,1);
  chart($("#uncertaintyChart"),state.metrics.history,[["model_uncertainty","#f4bf50"]],null,null);
}
setInterval(()=>{if($("#metricsDrawer").classList.contains("open"))loadMetrics()},3000);

function chart(el,data,series,minY,maxY){
  const c=el.getContext("2d"),dpr=devicePixelRatio,w=el.clientWidth*dpr,h=el.height*dpr;el.width=w;el.height=h;c.clearRect(0,0,w,h);
  c.strokeStyle="#2a2d36";c.lineWidth=1*dpr;
  for(let i=1;i<5;i++){const y=h*i/5;c.beginPath();c.moveTo(0,y);c.lineTo(w,y);c.stroke()}
  if(!data.length)return;
  const vals=data.flatMap(p=>series.map(s=>Number(p[s[0]])||0));
  const lo=minY??Math.min(...vals),hi=maxY??Math.max(...vals,lo+.001),span=Math.max(.0001,hi-lo);
  series.forEach(([key,color])=>{c.beginPath();data.forEach((p,i)=>{const x=i/(Math.max(1,data.length-1))*w,y=h-(Number(p[key])-lo)/span*h*.9-h*.05;(i?c.lineTo(x,y):c.moveTo(x,y))});c.strokeStyle=color;c.lineWidth=2*dpr;c.stroke()});
}
boot();
