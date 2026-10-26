const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const vm=require('node:vm');
const {execFileSync}=require('node:child_process');
const root=path.resolve(__dirname,'../../..');
const html=fs.readFileSync(path.join(root,'src/HMM/dashboard.html'),'utf8');
const script=html.match(/<script>([\s\S]*?)<\/script>/)[1];

function environment(){
  const elements=Object.fromEntries([...html.matchAll(/\bid="([^"]+)"/g)].map(m=>[m[1],{
    style:{},innerHTML:'',textContent:'',options:[],value:'15m',
  }]));
  const context=vm.createContext({console,Intl,Date,document:{getElementById(id){assert(elements[id],id);return elements[id];}}});
  vm.runInContext(script.slice(0,script.lastIndexOf('// \u2500\u2500 Init')),context);
  return {context,elements,run:source=>vm.runInContext(source,context)};
}
const identical={overlap_percent:100,jsd_bits:0,kl_bits:0,top1_agreement:1};
const disjoint={overlap_percent:0,jsd_bits:1,kl_bits:'Infinity',top1_agreement:0};
const close={overlap_percent:80,jsd_bits:0.1,kl_bits:0.3,top1_agreement:1};

test('closed-window metrics exclude open candles and retain pairwise coverage',()=>{
  const e=environment();
  e.context.view={times:[0,1,2,3],start:0,span:3,forecastClosed:[true,true,false,true],
    probabilityRows:[[identical,disjoint,close,close],[identical,null,close,disjoint],[identical,identical,close,identical]]};
  const result=e.run('windowForecastMetrics(view)');
  assert.deepEqual(Array.from(result.pairs,s=>s.n_compared),[3,2,3]);
  assert.deepEqual(Array.from(result.common,s=>s.n_compared),[2,2,2]);
  assert.equal(result.pairs[0].mean_kl_bits,Infinity);
  assert.equal(result.pairs[0].median_kl_bits,0.3);
  assert.equal(result.pairs[0].infinite_kl_count,1);
  e.context.view.start=1;e.context.view.span=0;
  assert.equal(e.run('windowForecastMetrics(view).pairs[0].n_compared'),1);
});

test('frontend means and medians match Python including infinite KL',()=>{
  const e=environment(),rows=[identical,disjoint,close];
  e.context.rows=rows;
  const actual=e.run('summarizeForecastMetrics(rows)');
  const python="import sys,json;sys.path.insert(0,'src/HMM');from forecast_comparison import summarize_comparisons,json_safe;print(json.dumps(json_safe(summarize_comparisons(json.loads(sys.argv[1])))))";
  const expected=JSON.parse(execFileSync('python',['-B','-c',python,JSON.stringify(rows)],{cwd:root,encoding:'utf8'}));
  for(const [key,value] of Object.entries(expected)){
    if(value==='Infinity') assert.equal(actual[key],Infinity);
    else assert(Math.abs(actual[key]-value)<1e-10,key);
  }
  e.context.rows=[{...identical,kl_bits:null}];
  assert.throws(()=>e.run('summarizeForecastMetrics(rows)'),/Invalid/);
});

test('probabilities align by candle timestamps, with an explicit common state order',()=>{
  const e=environment();
  const columns={overlap_percent:[100,0],jsd_bits:[0,1],kl_bits:[0,'Infinity'],top1_agreement:[1,0]};
  e.context.input={Q1:{price_timestamps:[1000,2000,3000,4000],close_prices:[1,2,3,4],timestamps:[2000,3000,4000],
    models:{A:[0,1,2],B:[0,1,2],C:[0,1,2],D:[0,1,2]},forecast_comparison:{
      state_order:['Low volatility','Medium volatility','High volatility'],timestamps:[3000,2000],closed:[true,false],
      pairs:{B:columns,C:columns,D:columns}}}};
  const output=e.run('comparisonData(input)');
  assert.equal(output.probabilityRows[0][0],null);
  assert.equal(output.probabilityRows[0][1].jsd_bits,1);
  assert.equal(output.probabilityRows[0][2].jsd_bits,0);
  assert.equal(output.probabilityRows[0][3],null);
  assert.deepEqual(Array.from(output.forecastClosed),[false,false,true,false]);
  e.context.input.Q1.forecast_comparison.state_order.reverse();
  assert.throws(()=>e.run('comparisonData(input)'),/Low, Medium, High/);
});

test('latest provisional metrics are distinct from window means',()=>{
  const e=environment();
  e.context.fixture={times:[0,1,2],states:[[0,0,0],[0,1,0],[0,0,1],[0,0,0]],start:0,span:2,mode:'live',
    forecastClosed:[true,true,false],probabilityRows:[[identical,disjoint,close],[identical,identical,close],[close,close,disjoint]]};
  e.context.forecasts=Object.fromEntries('ABCD'.split('').map(mid=>[mid,{origin_time:2000,target_time:3000,provisional:true,
    current_state:0,current_probability:0.8,next_state_probabilities:[0.6,0.3,0.1],tier_mean_rolling_vol:[0.001,0.002,0.004],
    comparison_with_A:close}]));
  e.run("comparisonView=fixture;updateForecastCards(forecasts,'15m');");
  const rendered=e.elements['comparison-agreement'].innerHTML;
  assert(rendered.includes('Mean JSD (bits)')&&rendered.includes('Mean KL (bits)'));
  assert(rendered.includes('50.0%')&&rendered.includes('2 / 3')&&rendered.includes('&infin;'));
  assert(rendered.includes('Latest next-candle forecast')&&rendered.includes('open candle'));
  assert(rendered.includes('80.0%'));
  assert.equal((rendered.match(/<th scope="row">A\u2013B<\/th>/g)||[]).length,2);
  assert(!/Window details|Median JSD|Median KL|Infinite KL \/ compared|High \/ non-High|rankings|Most to least similar/.test(rendered));
  assert(!rendered.includes('<th scope="row">B\u2013C</th>'));
  const before=e.run('windowForecastMetrics(comparisonView).pairs[0].mean_overlap_percent');
  e.context.forecasts.B.comparison_with_A=disjoint;
  e.run("updateForecastCards(forecasts,'15m');");
  assert.equal(e.run('windowForecastMetrics(comparisonView).pairs[0].mean_overlap_percent'),before);
});

test('2025 in-sample comparisons are labeled separately from live forecasts',()=>{
  const e=environment();
  e.run("comparisonView={times:[0],states:[[0],[0],[1],[2]],start:0,span:1,mode:'history'};updateComparisonAgreement();");
  const rendered=e.elements['comparison-agreement'].innerHTML;
  assert(rendered.includes('2025 in-sample comparison'));
  assert(!/Window details|Median JSD|Median KL|Infinite KL \/ compared|High \/ non-High|rankings|Most to least similar/.test(rendered));
  assert(!rendered.includes('Latest next-candle forecast'));
  assert(rendered.includes('100.0%')&&rendered.includes('Compared candles'));
  assert(!rendered.includes('Forecast candles'));
});

test('four price charts build and keep forecast metrics aligned through live/history switches',()=>{
  const e=environment();
  for(const element of Object.values(e.elements)){
    element.getContext=()=>({});
    element.classList={add(){},remove(){}};
  }
  e.context.cancelAnimationFrame=()=>{};
  e.context.Chart=class{constructor(ctx,config){Object.assign(this,config);}destroy(){}update(){}};
  const times=[0,1,2,3].map(i=>Date.UTC(2026,0,10)+i*900000);
  const column={overlap_percent:[100,100,100,100],jsd_bits:[0,0,0,0],kl_bits:[0,0,0,0],top1_agreement:[1,1,1,1]};
  const history={price_timestamps:times,close_prices:[100,101,100,102],timestamps:times,models:{A:[0,1,0,2],B:[0,1,0,2],C:[0,1,0,2],D:[0,1,0,2]}};
  e.context.input={Q1:{...history,forecast_comparison:{state_order:['Low volatility','Medium volatility','High volatility'],
    timestamps:times,closed:[true,true,true,false],pairs:{B:column,C:column,D:column}}}};
  e.run("buildComparisonCharts(input,'BTC','15m','live');");
  assert.equal(e.run('Object.keys(charts).length'),4);
  assert.equal(e.run('windowForecastMetrics(comparisonView).common[0].n_compared'),3);
  e.context.input={Q1:history};
  e.run("buildComparisonCharts(input,'BTC','15m','history');");
  assert.equal(e.run('Object.keys(charts).length'),4);
  assert.equal(e.run('windowForecastMetrics(comparisonView).pairs[0].n_compared'),0);
  assert(e.elements['comparison-agreement'].innerHTML.includes('2025 in-sample comparison'));
});

test('2025 is selected on first load and initial analysis does not request live data',async()=>{
  assert.match(html,/<option value="history" selected>2025 analysis<\/option>/);
  assert.doesNotMatch(html,/<option value="live" selected>/);
  const e=environment(),requests=[];
  e.elements['mode-sel'].value='history';
  e.context.document.querySelectorAll=()=>[];
  for(const element of Object.values(e.elements)) element.classList={add(){},remove(){}};
  e.context.fetch=async url=>{
    requests.push(url);
    return {ok:true,json:async()=>({coin:'BTC',tf:'15m',available:['Q1'],missing:[],
      results:[{model:'Model A',quarter:'Q1',log_likelihood:1}],regime_series:{Q1:{}}})};
  };
  e.run("renderAll=()=>{};runLive=()=>{throw new Error('Live must be optional');};");
  await e.run('runAnalysis()');
  assert.equal(requests.length,1);
  assert(requests[0].includes('/api/run?'));
  assert(e.elements['run-btn'].textContent.includes('Run 2025 Analysis'));
});

test('2025 quarterly distribution data populates all comparison statistics and counts',()=>{
  const e=environment();
  const times=[1000,2000,3000];
  const columns={overlap_percent:[80,90,100],jsd_bits:[0.2,0.1,0],kl_bits:[0.4,0.2,0],top1_agreement:[0,1,1]};
  e.context.input={Q1:{price_timestamps:times,close_prices:[100,101,102],timestamps:times,
    models:{A:[0,1,2],B:[0,1,2],C:[0,1,2],D:[0,1,2]},
    distribution_comparison:{method:'in_sample_quarter_fit',state_order:['Low volatility','Medium volatility','High volatility'],
      timestamps:times,closed:[true,true,true],pairs:{B:columns,C:columns,D:columns}}}};
  e.run("comparisonView={...comparisonData(input),start:1000,span:2000,mode:'history'};updateComparisonAgreement();");
  const summary=e.run('windowForecastMetrics(comparisonView).pairs[0]');
  assert.equal(summary.n_compared,3);
  assert.equal(summary.mean_overlap_percent,90);
  assert(Math.abs(summary.mean_jsd_bits-0.1)<1e-12);
  const rendered=e.elements['comparison-agreement'].innerHTML;
  assert(rendered.includes('90.0%')&&rendered.includes('0.1000')&&rendered.includes('0.2000'));
  assert(rendered.includes('66.7%')&&rendered.includes('Compared candles'));
  assert(rendered.includes('in-sample descriptions, not past-only forecasts'));
});

test('changing the coin automatically refreshes the default 2025 analysis',()=>{
  const e=environment();
  e.context.document.querySelectorAll=()=>[];
  e.context.started=0;
  e.run("loadFileMap=()=>{};runAnalysis=()=>{started++;};selectCoin('eth',true,{classList:{add(){}}});");
  assert.equal(e.context.started,1);
});
