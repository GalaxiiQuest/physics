"""
Energy model for thermal water extraction from lunar polar icy regolith.
Compares: baseline kiln, + spent-soil heat recuperation (known), and
+ condenser-to-feed latent heat recovery (the idea tested here).

Per 1 kg of water extracted. Thermal energy, ideal (no wall losses) lower bounds.
"""
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

T0 = 40.0          # K, feed temperature in a permanently shadowed region
L_SUB = 2.84e6     # J/kg, latent heat of sublimation of ice (~2.83-2.84 MJ/kg, 200-273 K)

def cp_reg(T):
    """Lunar soil heat capacity, Hemingway, Robie & Wilson (1973) fit, J/kg/K."""
    return -23.173 + 2.127*T + 1.5009e-2*T**2 - 7.3699e-5*T**3 + 9.6552e-8*T**4

def cp_ice(T):
    """Approximate linear fit to water-ice heat capacity (Giauque & Stout), J/kg/K."""
    return 185.0 + 7.037*T

def dH_reg(T1, T2):
    return quad(cp_reg, T1, T2)[0]

def dH_ice(T1, T2):
    return quad(cp_ice, T1, T2)[0]

def energy(w, Tx=250.0, Tc=220.0, dT=20.0, eps=0.0, cond_recovery=False, yield_=1.0):
    """
    w: ice mass fraction of icy soil; Tx: extraction temperature (K);
    Tc: condenser temperature (K); dT: heat-exchanger approach (K);
    eps: spent-soil recuperator effectiveness; cond_recovery: route condenser heat into feed.
    yield_: fraction of ice actually extracted (unextracted ice still gets heated).
    Returns dict, energies in kWh per kg of water collected.
    """
    ice_in = 1.0 / yield_                  # kg ice processed per kg water collected
    m_r = ice_in * (1 - w) / w             # kg dry regolith processed
    ice_left = ice_in - 1.0                # ice left in spent soil (heated, not collected)

    # enthalpy of feed (soil + ice) from T0 to temperature T
    def feed_rise(T):
        return m_r*dH_reg(T0, T) + ice_in*dH_ice(T0, T)

    Q_feed_total = feed_rise(Tx)           # heat feed 40 K -> Tx
    Q_lat = 1.0 * L_SUB                    # sublimate collected water
    Q_lat_lost = ice_left * L_SUB * 0.0    # assume unextracted ice does not sublimate

    # 1) condenser heat routed into the cold feed (indirect exchanger)
    Q_c_avail = L_SUB                      # deposition heat released at condenser per kg water
    Q_c_used, T1 = 0.0, T0
    if cond_recovery:
        cap = feed_rise(Tc - dT)
        Q_c_used = min(Q_c_avail, cap)
        T1 = brentq(lambda T: feed_rise(T) - Q_c_used, T0, Tc - dT + 1e-9) if Q_c_used > 0 else T0
    Q_rad = Q_c_avail - Q_c_used           # heat that must still be radiated away

    # 2) spent-soil counterflow recuperator (hot dry soil at Tx -> feed at T1)
    Q_rec = 0.0
    if eps > 0:
        hot_max = m_r*dH_reg(T1, Tx) + ice_left*dH_ice(T1, Tx)   # spent soil can cool to T1
        cold_max = feed_rise(Tx) - feed_rise(T1)                  # feed can warm to Tx
        Q_rec = eps * min(hot_max, cold_max)

    Q_ext = Q_feed_total + Q_lat - Q_c_used - Q_rec
    k = 1/3.6e6
    return dict(w=w, m_r=m_r, Q_ext=Q_ext*k, Q_sens=Q_feed_total*k, Q_lat=Q_lat*k,
                Q_cond_used=Q_c_used*k, Q_rec=Q_rec*k, Q_rad=Q_rad*k, T_pre=T1)

def crossover(Tc=220.0, dT=20.0):
    """Ice fraction below which the feed can absorb ALL condenser heat (no radiator)."""
    f = lambda w: energy(w, Tc=Tc, dT=dT, cond_recovery=True)['Q_rad'] - 1e-6
    # Q_rad is ~0 below crossover; find where feed capacity == latent heat
    def g(w):
        m_r = (1 - w) / w
        return m_r*dH_reg(T0, Tc - dT) + dH_ice(T0, Tc - dT) - L_SUB
    return brentq(g, 0.005, 0.9)

if __name__ == "__main__":
    print("cp_reg at 40, 100, 200, 250, 300 K:",
          [round(cp_reg(T)) for T in (40, 100, 200, 250, 300)])
    print("Regolith enthalpy 40->200 K: %.1f kJ/kg" % (dH_reg(40, 200)/1e3))
    print("Regolith enthalpy 40->250 K: %.1f kJ/kg" % (dH_reg(40, 250)/1e3))
    print()
    cases = [
        ("A baseline kiln (Tx=250K)",          dict(eps=0.0, cond_recovery=False)),
        ("B + spent-soil recuperator eps=0.7", dict(eps=0.7, cond_recovery=False)),
        ("C + condenser->feed only",           dict(eps=0.0, cond_recovery=True)),
        ("D + both",                           dict(eps=0.7, cond_recovery=True)),
    ]
    print("%-38s" % "case", "  ".join("w=%2d%%" % (w*100) for w in (0.01, 0.02, 0.05, 0.10)))
    for name, kw in cases:
        row = [energy(w, **kw)['Q_ext'] for w in (0.01, 0.02, 0.05, 0.10)]
        print("%-38s" % name, "  ".join("%5.2f" % r for r in row))
    print("\n(kWh thermal per kg water, ideal lower bounds)")
    print()
    for w in (0.01, 0.02, 0.05, 0.10):
        e = energy(w, cond_recovery=True)
        print("w=%4.0f%%: condenser heat used %.3f of %.3f kWh/kg; radiator still needs %.3f; feed preheated to %.0f K"
              % (w*100, e['Q_cond_used'], e['Q_lat'], e['Q_rad'], e['T_pre']))
    for Tc in (180, 200, 220, 240):
        print("Crossover ice fraction (no radiator needed) at Tc=%d K: %.1f%%" % (Tc, crossover(Tc)*100))
    print()
    # extraction temperature sensitivity (baseline vs D)
    for Tx in (250, 300, 373, 473):
        a = energy(0.05, Tx=Tx)['Q_ext']; d = energy(0.05, Tx=Tx, eps=0.7, cond_recovery=True)['Q_ext']
        print("Tx=%d K, w=5%%: baseline %.2f, with both %.2f kWh/kg (%.0f%% cut)" % (Tx, a, d, 100*(1-d/a)))
    print()
    # yield sensitivity
    for y in (1.0, 0.8, 0.6):
        a = energy(0.05, yield_=y)['Q_ext']; d = energy(0.05, yield_=y, eps=0.7, cond_recovery=True)['Q_ext']
        print("yield %.0f%%, w=5%%: baseline %.2f, both %.2f" % (y*100, a, d))
    print()
    # city-scale radiator area avoided: 15,000 t/yr
    mdot = 15e6 / (365.25*86400)
    Prad = mdot * L_SUB
    sigma = 5.670e-8
    for Tc in (200, 220):
        A = Prad / (0.9*sigma*Tc**4)
        print("City scale: %.2f MW condenser heat; ideal radiator at %d K needs %.0f m^2" % (Prad/1e6, Tc, A))
