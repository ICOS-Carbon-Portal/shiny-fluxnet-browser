(function () {
  var CONSENT_KEY = 'icos_consent_v1';
  if (localStorage.getItem(CONSENT_KEY) === '1') return;

  function _showConsent() {
    var backdrop = document.createElement('div');
    backdrop.id = 'icos-consent-backdrop';
    backdrop.style.cssText = [
      'position:fixed', 'top:0', 'left:0', 'width:100%', 'height:100%',
      'background:rgba(0,0,0,0.65)', 'z-index:99999',
      'display:flex', 'align-items:center', 'justify-content:center'
    ].join(';');

    backdrop.innerHTML = [
      '<div style="background:#fff;border-radius:8px;padding:1.8rem 2rem;max-width:520px;',
      'width:90%;box-shadow:0 4px 28px rgba(0,0,0,0.35);">',
      '<h5 style="font-size:13pt;margin-bottom:0.8rem;">Welcome to the ICOS FLUXNET Data Browser</h5>',
      '<p style="font-size:10pt;margin-bottom:1rem;">',
      'Before continuing, please read and accept the terms below. This application stores your',
      ' plot configurations in your browser\'s local storage for an optimal experience.',
      '</p>',
      '<div style="margin-bottom:0.6rem;">',
      '<label style="display:flex;align-items:flex-start;gap:0.5rem;cursor:pointer;font-size:10pt;font-weight:normal;">',
      '<input type="checkbox" id="icos_cb1" style="margin-top:3px;flex-shrink:0;">',
      '<span>I have read and accept the',
      ' <a href="https://www.icos-cp.eu/privacy" target="_blank" rel="noopener">ICOS Privacy Licence</a>',
      '</span></label></div>',
      '<div style="margin-bottom:0.6rem;">',
      '<label style="display:flex;align-items:flex-start;gap:0.5rem;cursor:pointer;font-size:10pt;font-weight:normal;">',
      '<input type="checkbox" id="icos_cb2" style="margin-top:3px;flex-shrink:0;">',
      '<span>I have read and accept the',
      ' <a href="https://www.icos-cp.eu/terms-of-use" target="_blank" rel="noopener">Terms of Use</a>',
      '</span></label></div>',
      '<div style="margin-bottom:1.4rem;">',
      '<label style="display:flex;align-items:flex-start;gap:0.5rem;cursor:pointer;font-size:10pt;font-weight:normal;">',
      '<input type="checkbox" id="icos_cb3" style="margin-top:3px;flex-shrink:0;">',
      '<span>I have read and accept the',
      ' <a href="https://www.icos-cp.eu/privacy" target="_blank" rel="noopener">Privacy Policy</a>',
      '</span></label></div>',
      '<button id="icos_accept_btn" disabled',
      ' style="width:100%;padding:0.45rem;background:#00ABC9;color:#fff;border:none;',
      'border-radius:4px;font-size:10pt;opacity:0.45;cursor:default;">',
      'Accept &amp; Continue</button>',
      '</div>'
    ].join('');

    document.body.appendChild(backdrop);

    var btn = document.getElementById('icos_accept_btn');
    var cbs = ['icos_cb1', 'icos_cb2', 'icos_cb3'].map(function (id) {
      return document.getElementById(id);
    });

    function sync() {
      var all = cbs.every(function (cb) { return cb.checked; });
      btn.disabled = !all;
      btn.style.opacity = all ? '1' : '0.45';
      btn.style.cursor = all ? 'pointer' : 'default';
    }

    cbs.forEach(function (cb) { cb.addEventListener('change', sync); });

    btn.addEventListener('click', function () {
      localStorage.setItem(CONSENT_KEY, '1');
      backdrop.remove();
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', _showConsent);
  } else {
    _showConsent();
  }
}());
