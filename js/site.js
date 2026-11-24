(() => {
  'use strict';
  document.body.classList.add('js-ready');
  const menu = document.querySelector('.menu-toggle');
  const nav = document.querySelector('.site-nav');
  menu.hidden = false;
  const closeMenu = () => {
    menu.setAttribute('aria-expanded', 'false');
    menu.setAttribute('aria-label', 'Open navigation');
    nav.classList.remove('is-open');
  };
  menu.addEventListener('click', () => {
    const open = menu.getAttribute('aria-expanded') !== 'true';
    menu.setAttribute('aria-expanded', String(open));
    menu.setAttribute('aria-label', open ? 'Close navigation' : 'Open navigation');
    nav.classList.toggle('is-open', open);
  });
  nav.querySelectorAll('a').forEach(link => link.addEventListener('click', closeMenu));
  document.addEventListener('keydown', event => {
    if (event.key === 'Escape' && menu.getAttribute('aria-expanded') === 'true') {
      closeMenu();
      menu.focus();
    }
  });
  document.addEventListener('click', event => {
    if (!event.target.closest('.site-header')) closeMenu();
  });
  matchMedia('(min-width: 901px)').addEventListener('change', closeMenu);

  const dialog = document.querySelector('#figure-dialog');
  document.querySelectorAll('[data-figure]').forEach(link => {
    link.addEventListener('click', event => {
      if (typeof dialog.showModal !== 'function') return;
      event.preventDefault();
      const image = document.querySelector('#figure-image');
      image.src = link.href;
      image.alt = link.querySelector('img').alt;
      document.querySelector('#figure-caption').textContent = link.dataset.caption;
      dialog.showModal();
      document.body.classList.add('dialog-open');
    });
  });
  dialog.querySelector('.dialog-close').addEventListener('click', () => dialog.close());
  dialog.addEventListener('close', () => document.body.classList.remove('dialog-open'));
  dialog.addEventListener('click', event => {
    const rect = dialog.getBoundingClientRect();
    if (event.target === dialog && (event.clientX < rect.left || event.clientX > rect.right ||
        event.clientY < rect.top || event.clientY > rect.bottom)) dialog.close();
  });

  if (!window.P2G_DATA) {
    document.querySelectorAll('.encoder-chart, .results-tables').forEach(chart => {
      chart.textContent = 'Data could not be loaded. Download the CSV below.';
    });
    return;
  }
  const data = window.P2G_DATA;
  const format = value => value.toFixed(4);
  const metricNames = { pcc200: 'PCC-200', pcc50: 'PCC-50', mse: 'MSE', mae: 'MAE' };
  let encoderDataset = 'Brain';
  let resultsPanel = 'HMHVG';

  function element(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function methodLabel(row) {
    const label = element('span', 'method-name', row.method);
    if (row.adapted) label.append(element('sup', '', '†'));
    return label;
  }

  const hasPairedSD = row => row.n_pairs === 3 &&
    [row.reference_sd, row.virchow2_sd].every(sd => Number.isFinite(sd) && sd >= 0);
  const improved = data.swaps.filter(row => row.virchow2 > row.reference).length;
  document.querySelector('#encoder-summary').replaceChildren(
    'Virchow2 improves ', element('strong', '', `${improved} of ${data.swaps.length}`),
    ' dataset–method–panel pairs.');

  function renderEncoder() {
    const chart = document.querySelector('#encoder-chart');
    chart.replaceChildren();
    chart.setAttribute('aria-label', `${encoderDataset}, HMHVG-200 and HVG-200, both prediction settings`);
    document.querySelector('#encoder-title').textContent = encoderDataset;
    const rows = data.swaps.filter(row => row.dataset === encoderDataset);
    // Share one scale for the plotted means across all methods and panels.
    const endpoints = rows.flatMap(row => [row.reference, row.virchow2]);
    const low = Math.min(...endpoints);
    const high = Math.max(...endpoints);
    const padding = Math.max(high - low, .1) * .06;
    const step = [.02, .05, .1, .2, .5, 1, 2].find(value => value >= (high - low + 2 * padding) / 6) || 2;
    const firstTick = Math.floor((low - padding) / step);
    const lastTick = Math.ceil((high + padding) / step);
    const axisMin = firstTick * step;
    const axisMax = lastTick * step;
    const position = value => (value - axisMin) / (axisMax - axisMin) * 100;
    const ticks = document.querySelector('#encoder-axis-ticks');
    ticks.replaceChildren();
    for (let tick = firstTick; tick <= lastTick; tick++) {
      ticks.append(element('span', '', (tick * step).toFixed(step < .1 ? 2 : 1)));
    }
    const groups = new Map();
    for (const row of rows) {
      const groupKey = `${row.regime}:${row.method}`;
      if (!groups.has(groupKey)) {
        const group = element('section', 'encoder-group');
        group.setAttribute('aria-label', `${row.method}, ${row.regime}-patch`);
        const label = element('h4', 'encoder-label');
        label.append(methodLabel(row), element('span', 'setting-tag', `${row.regime === 'single' ? 'Single' : 'Cross'}-patch`),
          element('small', '', data.reference_inputs[row.method]));
        group.append(label);
        groups.set(groupKey, group);
        chart.append(group);
      }
      const pairedSD = hasPairedSD(row);
      const item = element('div', 'encoder-row');
      item.dataset.panel = row.panel;
      const label = element('div', 'encoder-panel-label', `${row.panel}-200`);
      const plot = element('div', 'paired-points');
      plot.setAttribute('aria-hidden', 'true');
      plot.style.setProperty('--tick-spacing', `${100 / (lastTick - firstTick)}%`);
      const delta = row.virchow2 - row.reference;
      const connector = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
      connector.classList.add('pair-connector');
      const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      line.classList.add('pair-line');
      if (delta < 0) line.classList.add('is-negative');
      for (const [key, value] of Object.entries({
        x1: `${position(row.reference)}%`, x2: `${position(row.virchow2)}%`, y1: '50%', y2: '50%'
      })) line.setAttribute(key, value);
      connector.append(line);
      plot.append(connector);
      for (const [name, value] of [['reference', row.reference], ['virchow', row.virchow2]]) {
        const point = element('span', `pair-point ${name}-point`);
        if (name === 'virchow' && delta < 0) point.classList.add('is-negative');
        point.style.left = `${position(value)}%`;
        plot.append(point);
      }
      const values = element('div', 'encoder-values');
      const scores = element('div', 'encoder-scores');
      for (const [name, title] of [['reference', 'Reference input'], ['virchow2', 'Virchow2']]) {
        const score = element('span', 'encoder-score');
        const key = element('i', name === 'reference' ? 'reference-dot' : 'virchow-dot');
        key.setAttribute('aria-hidden', 'true');
        score.append(key, element('span', 'visually-hidden', `${title}: `), format(row[name]));
        if (pairedSD) score.append(element('small', 'encoder-score-sd', ` ± ${format(row[`${name}_sd`])}`));
        scores.append(score);
      }
      values.append(scores);
      const difference = element('strong', delta < 0 ? 'is-negative' : '', `${delta >= 0 ? '+' : '−'}${format(Math.abs(delta))}`);
      if (pairedSD && Number.isFinite(row.delta_sd) && row.delta_sd >= 0) {
        difference.append(element('small', 'delta-sd', ` ± ${format(row.delta_sd)}`));
      }
      difference.append(element('small', '', ' ΔPCC'));
      values.append(difference);
      item.append(label, plot, values);
      groups.get(groupKey).append(item);
    }
  }

  document.querySelectorAll('[data-encoder-dataset]').forEach(button => button.addEventListener('click', () => {
    encoderDataset = button.dataset.encoderDataset;
    document.querySelectorAll('[data-encoder-dataset]').forEach(item =>
      item.setAttribute('aria-pressed', String(item === button)));
    renderEncoder();
  }));
  renderEncoder();

  function renderResults() {
    const metric = document.querySelector('#results-metric').value;
    const higher = metric.startsWith('pcc');
    const metricLabel = `${metricNames[metric]} ${higher ? '↑' : '↓'}`;
    for (const regime of ['single', 'cross']) {
      const table = document.querySelector(`#results-${regime}`);
      table.replaceChildren();
      const caption = element('caption', 'visually-hidden', `${regime === 'single' ? 'Single' : 'Cross'}-patch, ${resultsPanel}-200, ${metricLabel}. Mean and sample SD over three held-out slides. Bold marks the best mean in each column.`);
      const header = element('thead');
      const headerRow = element('tr');
      for (const title of ['Method', ...data.datasets]) {
        const cell = element('th', '', title);
        cell.scope = 'col';
        headerRow.append(cell);
      }
      header.append(headerRow);
      const rows = data.methods[regime].map(method => {
        const records = data.datasets.map(dataset => data.results.find(row =>
          row.regime === regime && row.panel === resultsPanel && row.dataset === dataset && row.method === method));
        const values = records.map(row => row[metric]);
        return { method, records, values };
      });
      const best = Array.from({ length: data.datasets.length }, (_, column) => {
        const values = rows.map(row => Number(row.values[column].toFixed(4)));
        return values.length ? (higher ? Math.max(...values) : Math.min(...values)) : null;
      });
      const body = element('tbody');
      for (const row of rows) {
        const ours = row.method.startsWith('SimpleP2G');
        const tr = element('tr', ours ? 'is-ours' : '');
        tr.dataset.method = row.method;
        const name = element('th');
        name.scope = 'row';
        name.append(methodLabel(row.records[0]));
        if (ours) name.append(element('small', 'ours-label', 'Ours'));
        tr.append(name);
        row.values.forEach((value, column) => {
          const cell = element('td');
          cell.dataset.column = data.datasets[column];
          const winner = Number(value.toFixed(4)) === best[column];
          cell.append(element(winner ? 'strong' : 'span', 'score', format(value)));
          if (winner) cell.classList.add('is-best');
          cell.append(element('small', 'score-sd', `± ${format(row.records[column][`${metric}_sd`])}`));
          tr.append(cell);
        });
        body.append(tr);
      }
      table.append(caption, header, body);
    }
  }
  document.querySelectorAll('[data-results-panel]').forEach(button => button.addEventListener('click', () => {
    resultsPanel = button.dataset.resultsPanel;
    document.querySelectorAll('[data-results-panel]').forEach(item =>
      item.setAttribute('aria-pressed', String(item === button)));
    renderResults();
  }));
  document.querySelector('#results-metric').addEventListener('change', renderResults);
  renderResults();

  if ('IntersectionObserver' in window) {
    const links = [...nav.querySelectorAll('a[href^="#"]')];
    const observer = new IntersectionObserver(entries => {
      for (const entry of entries) {
        if (!entry.isIntersecting) continue;
        links.forEach(link => {
          if (link.hash === `#${entry.target.id}`) link.setAttribute('aria-current', 'location');
          else link.removeAttribute('aria-current');
        });
      }
    }, { rootMargin: '-15% 0px -65% 0px' });
    observer.observe(document.querySelector('#top'));
    links.forEach(link => observer.observe(document.querySelector(link.hash)));
  }
})();
