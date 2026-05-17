import { useState } from 'react';
import './index.css';

function App() {
  const [date, setDate] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [activeDiagnosticsTab, setActiveDiagnosticsTab] = useState('features');

  const handlePredict = async (e) => {
    e.preventDefault();
    if (!date) {
      setError('Please select a date first.');
      return;
    }

    setLoading(true);
    setError('');

    try {
      const apiBaseUrl = import.meta.env.VITE_API_URL || 'http://localhost:5000';
      const response = await fetch(`${apiBaseUrl}/predict?date=${date}`);
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Failed to fetch prediction');
      }

      setResult(data);
    } catch (err) {
      setError(err.message);
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  // Get professional advice based on AQI category
  const getHealthAdvice = (category) => {
    switch (category) {
      case 'Good':
        return {
          title: 'Excellent Air Quality',
          text: 'Air quality is considered satisfactory, and air pollution poses little or no risk. Perfect time for outdoor exercises and recreational activities.',
          icon: '☀️'
        };
      case 'Satisfactory':
        return {
          title: 'Satisfactory Air Quality',
          text: 'Air quality is acceptable. However, minor health effects may occur for extremely sensitive individuals. Enjoy the day outdoors!',
          icon: '🌤️'
        };
      case 'Moderate':
        return {
          title: 'Moderate Health Alert',
          text: 'Sensitive individuals might experience minor breathing discomfort. Consider reducing intense outdoor workouts if you experience fatigue or irritation.',
          icon: '😷'
        };
      case 'Poor':
        return {
          title: 'Active Pollution Warning',
          text: 'May cause breathing discomfort to most people on prolonged exposure. People with heart or lung diseases should limit prolonged outdoor exertion.',
          icon: '⚠️'
        };
      case 'Very Poor':
        return {
          title: 'Severe Health Precaution',
          text: 'Significant health risk for everyone. Everyone may experience respiratory discomfort on prolonged exposure. Avoid strenuous outdoor activities; keep windows closed.',
          icon: '🚨'
        };
      case 'Severe':
        return {
          title: 'Emergency Health Hazard',
          text: 'Extremely critical condition. Serious respiratory impact even on healthy people. Stay indoors, run air purifiers, and wear certified N95 masks if outdoor travel is essential.',
          icon: '❌'
        };
      default:
        return {
          title: 'Unknown Air Condition',
          text: 'No guidance available.',
          icon: '❓'
        };
    }
  };

  // Helper to render inline weather icons
  const getWeatherIcon = (type) => {
    switch (type) {
      case 'temp':
        return (
          <svg className="w-icon" viewBox="0 0 24 24" fill="none" stroke="#f43f5e" strokeWidth="2">
            <path d="M14 14.76V3.5a2.5 2.5 0 0 0-5 0v11.26a4.5 4.5 0 1 0 5 0z" />
          </svg>
        );
      case 'humidity':
        return (
          <svg className="w-icon" viewBox="0 0 24 24" fill="none" stroke="#0ea5e9" strokeWidth="2">
            <path d="M12 22a7 7 0 0 0 7-7c0-4.3-7-13-7-13S5 11 5 15a7 7 0 0 0 7 7z" />
          </svg>
        );
      case 'wind':
        return (
          <svg className="w-icon" viewBox="0 0 24 24" fill="none" stroke="#10b981" strokeWidth="2">
            <path d="M9.59 4.59A2 2 0 1 1 11 8H2m10.59 11.41A2 2 0 1 0 14 16H2m15.73-8.27A2.5 2.5 0 1 1 19.5 12H2" />
          </svg>
        );
      case 'precip':
        return (
          <svg className="w-icon" viewBox="0 0 24 24" fill="none" stroke="#6366f1" strokeWidth="2">
            <path d="M20 17.58A5 5 0 0 0 18 8h-1.26A8 8 0 1 0 4 15.25" />
            <path d="M8 16v4M12 18v4M16 16v4" />
          </svg>
        );
      default:
        return null;
    }
  };

  // Draw pure SVG chart for trend
  const renderTrendChart = (trend) => {
    if (!trend || trend.length === 0) return null;

    const width = 500;
    const height = 100;
    const padding = 15;

    // Find min and max values for dynamic scaling
    const aqiValues = trend.map(t => t.aqi);
    const maxVal = Math.max(...aqiValues, 300); // Scale up to at least 300
    const minVal = Math.min(...aqiValues, 50);

    const getX = (index) => padding + (index * (width - 2 * padding)) / (trend.length - 1);
    const getY = (value) => height - padding - ((value - minVal) * (height - 2 * padding)) / (maxVal - minVal || 1);

    const points = trend.map((t, idx) => `${getX(idx)},${getY(t.aqi)}`).join(' ');

    return (
      <div className="trend-chart-wrapper">
        <svg viewBox={`0 0 ${width} ${height}`} className="sparkline-svg">
          {/* Grid lines */}
          <line x1={padding} y1={getY(100)} x2={width - padding} y2={getY(100)} className="chart-grid-line" strokeDasharray="3,3" />
          <line x1={padding} y1={getY(200)} x2={width - padding} y2={getY(200)} className="chart-grid-line" strokeDasharray="3,3" />
          
          {/* Trend line */}
          <polyline
            fill="none"
            stroke="url(#chartGrad)"
            strokeWidth="3.5"
            strokeLinecap="round"
            strokeLinejoin="round"
            points={points}
            className="sparkline-line"
          />

          {/* Define beautiful gradient */}
          <defs>
            <linearGradient id="chartGrad" x1="0" y1="0" x2="1" y2="0">
              <stop offset="0%" stopColor="#3b82f6" />
              <stop offset="100%" stopColor="#6366f1" />
            </linearGradient>
          </defs>

          {/* Interactive dots */}
          <g className="sparkline-dots">
            {trend.map((t, idx) => (
              <circle
                key={idx}
                cx={getX(idx)}
                cy={getY(t.aqi)}
                r="5"
                fill={t.color}
                stroke="#ffffff"
                strokeWidth="2.5"
                boxShadow="0 4px 6px rgba(0,0,0,0.1)"
              />
            ))}
          </g>
        </svg>
        <div className="chart-labels">
          {trend.map((t, idx) => {
            const shortDate = t.date.split('-').slice(1).join('/'); // MM/DD
            return (
              <div key={idx} className="chart-label-item">
                <div>{shortDate}</div>
                <div style={{ color: t.color, fontWeight: 700 }}>{Math.round(t.aqi)}</div>
              </div>
            );
          })}
        </div>
      </div>
    );
  };

  const renderNormalCurve = (res) => {
    const pdf = res.stats.pdf;
    if (!pdf || pdf.length === 0) return null;
    
    const minX = pdf[0].x;
    const maxX = pdf[pdf.length - 1].x;
    const maxY = Math.max(...pdf.map(p => p.y));
    
    const svgW = 320;
    const svgH = 140;
    const paddingLeft = 30;
    const paddingRight = 10;
    const paddingTop = 25;
    const paddingBottom = 20;
    
    const points = pdf.map(p => {
      const xRatio = (p.x - minX) / (maxX - minX);
      const yRatio = p.y / maxY;
      const svgX = paddingLeft + xRatio * (svgW - paddingLeft - paddingRight);
      const svgY = svgH - paddingBottom - yRatio * (svgH - paddingTop - paddingBottom);
      return `${svgX},${svgY}`;
    });
    
    const pathData = `M ${points.join(' L ')}`;
    const closedPathData = `${pathData} L ${svgW - paddingRight},${svgH - paddingBottom} L ${paddingLeft},${svgH - paddingBottom} Z`;
    
    const predRatio = (res.predicted_aqi - minX) / (maxX - minX);
    const boundedRatio = Math.max(0, Math.min(1, predRatio));
    const predSvgX = paddingLeft + boundedRatio * (svgW - paddingLeft - paddingRight);
    
    return (
      <div className="normal-curve-svg-wrapper">
        <svg viewBox={`0 0 ${svgW} ${svgH}`} className="normal-curve-svg">
          <defs>
            <linearGradient id="pdfGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="var(--primary)" stopOpacity="0.25" />
              <stop offset="100%" stopColor="var(--primary)" stopOpacity="0.0" />
            </linearGradient>
          </defs>
          <line x1={paddingLeft} y1={svgH - paddingBottom} x2={svgW - paddingRight} y2={svgH - paddingBottom} stroke="var(--border-medium)" strokeWidth="1.5" />
          <path d={closedPathData} fill="url(#pdfGrad)" />
          <path d={pathData} fill="none" stroke="var(--primary)" strokeWidth="3" strokeLinecap="round" />
          <line x1={predSvgX} y1={5} x2={predSvgX} y2={svgH - paddingBottom} stroke="#be123c" strokeWidth="2" strokeDasharray="4 3" />
          <circle cx={predSvgX} cy={5} r="3" fill="#be123c" />
          <text x={predSvgX} y={15} fill="#be123c" fontSize="9" fontWeight="800" textAnchor={predSvgX > svgW / 2 ? 'end' : 'start'} dx={predSvgX > svgW / 2 ? -6 : 6}>Predicted ({res.predicted_aqi})</text>
          <text x={paddingLeft} y={svgH - 5} fill="var(--text-muted)" fontSize="8" fontWeight="600" textAnchor="middle">{Math.round(minX)}</text>
          <text x={svgW - paddingRight} y={svgH - 5} fill="var(--text-muted)" fontSize="8" fontWeight="600" textAnchor="middle">{Math.round(maxX)}</text>
          <text x={(svgW - paddingLeft - paddingRight) / 2 + paddingLeft} y={svgH - 5} fill="var(--text-muted)" fontSize="8" fontWeight="700" textAnchor="middle">Mean ({Math.round(res.stats.mean)})</text>
        </svg>
      </div>
    );
  };

  const advice = result ? getHealthAdvice(result.category) : null;

  return (
    <div className="dashboard">
      {/* Left Panel */}
      <div className="left-panel">
        <div className="brand-section">
          <span className="brand-badge">AQI Intelligent Engine</span>
          <h1>Delhi Air Quality Dashboard</h1>
          <p>Real-time prediction utilizing an advanced two-stage machine learning ensemble model (LightGBM, XGBoost, and Ridge Regressors).</p>
        </div>

        <div className="card">
          <h2 className="form-title">Enter Target Date</h2>
          <form onSubmit={handlePredict}>
            <div className="input-container">
              <label htmlFor="date" className="input-label">Select Calendar Date</label>
              <input
                type="date"
                id="date"
                className="date-input"
                value={date}
                onChange={(e) => setDate(e.target.value)}
                min="2017-01-08"
              />
            </div>
            <button type="submit" className="btn-predict" disabled={loading}>
              {loading ? <div className="spinner" style={{ width: '20px', height: '20px', borderWidth: '2px' }}></div> : 'Calculate Prediction'}
            </button>
          </form>
        </div>

        <div className="card info-panel">
          <h3 style={{ fontSize: '1rem', fontWeight: 800 }}>MODEL HIGHLIGHTS</h3>
          <div className="info-item">
            <div className="info-icon">🧠</div>
            <div className="info-text">
              <h4>Chronological Isolation</h4>
              <p>Prevents time-series leakage, ensuring exact validation accuracies.</p>
            </div>
          </div>
          <div className="info-item">
            <div className="info-icon">📊</div>
            <div className="info-text">
              <h4>Asymmetric Error Weights</h4>
              <p>Model is penalized 2x harder for dangerous under-prediction.</p>
            </div>
        </div>
      </div>

      <div className="card diagnostics-panel">
        <div className="diagnostics-header">
          <h3>🧠 AI DIAGNOSTICS & CALIBRATION</h3>
          <div className="tab-buttons">
            <button 
              type="button"
              className={`tab-btn ${activeDiagnosticsTab === 'features' ? 'active' : ''}`}
              onClick={() => setActiveDiagnosticsTab('features')}
            >
              Drivers
            </button>
            <button 
              type="button"
              className={`tab-btn ${activeDiagnosticsTab === 'stats' ? 'active' : ''}`}
              onClick={() => setActiveDiagnosticsTab('stats')}
            >
              Stats
            </button>
            <button 
              type="button"
              className={`tab-btn ${activeDiagnosticsTab === 'distribution' ? 'active' : ''}`}
              onClick={() => setActiveDiagnosticsTab('distribution')}
            >
              Distribution
            </button>
          </div>
        </div>

        {activeDiagnosticsTab === 'features' && (
          <div className="features-list">
            <p className="tab-desc">Top 5 features driving the Two-Stage ML Ensemble:</p>
            
            <div className="feature-item">
              <div className="feature-info">
                <span className="feature-name">AQI Yesterday <code className="feature-code">AQI_lag_1</code></span>
                <span className="feature-weight">32.40%</span>
              </div>
              <div className="progress-bar-container">
                <div className="progress-bar-fill" style={{ width: '32.4%' }}></div>
              </div>
            </div>

            <div className="feature-item">
              <div className="feature-info">
                <span className="feature-name">7-Day Exp Average <code className="feature-code">AQI_ewma_7</code></span>
                <span className="feature-weight">27.31%</span>
              </div>
              <div className="progress-bar-container">
                <div className="progress-bar-fill" style={{ width: '27.31%' }}></div>
              </div>
            </div>

            <div className="feature-item">
              <div className="feature-info">
                <span className="feature-name">Seasonal Cosine Shift <code className="feature-code">Month_cos</code></span>
                <span className="feature-weight">7.67%</span>
              </div>
              <div className="progress-bar-container">
                <div className="progress-bar-fill" style={{ width: '7.67%' }}></div>
              </div>
            </div>

            <div className="feature-item">
              <div className="feature-info">
                <span className="feature-name">Annual Cyclical Factor <code className="feature-code">DayOfYear_cos</code></span>
                <span className="feature-weight">6.19%</span>
              </div>
              <div className="progress-bar-container">
                <div className="progress-bar-fill" style={{ width: '6.19%' }}></div>
              </div>
            </div>

            <div className="feature-item">
              <div className="feature-info">
                <span className="feature-name">7-Day Roll Mean <code className="feature-code">AQI_roll_mean_7</code></span>
                <span className="feature-weight">3.34%</span>
              </div>
              <div className="progress-bar-container">
                <div className="progress-bar-fill" style={{ width: '3.34%' }}></div>
              </div>
            </div>

            <div className="calibration-table-container" style={{ marginTop: '1rem', borderTop: '1px solid var(--border-light)', paddingTop: '1.25rem' }}>
              <p className="tab-desc" style={{ marginBottom: '0.6rem' }}>Model Classification Precision (Validation split):</p>
              <table className="calibration-table">
                <thead>
                  <tr>
                    <th>Alert Level</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1-Score</th>
                    <th>Support</th>
                  </tr>
                </thead>
                <tbody>
                  <tr className="cal-row satisfactory-row">
                    <td><span className="cal-dot" style={{ backgroundColor: '#facc15' }}></span>Satisfactory</td>
                    <td>0.00</td>
                    <td>0.00</td>
                    <td>0.00</td>
                    <td>2</td>
                  </tr>
                  <tr className="cal-row moderate-row">
                    <td><span className="cal-dot" style={{ backgroundColor: '#fb923c' }}></span>Moderate</td>
                    <td>0.86</td>
                    <td>0.81</td>
                    <td>0.83</td>
                    <td>53</td>
                  </tr>
                  <tr className="cal-row poor-row">
                    <td><span className="cal-dot" style={{ backgroundColor: '#f87171' }}></span>Poor</td>
                    <td>0.75</td>
                    <td>0.79</td>
                    <td>0.77</td>
                    <td>56</td>
                  </tr>
                  <tr className="cal-row verypoor-row">
                    <td><span className="cal-dot" style={{ backgroundColor: '#a855f7' }}></span>Very Poor</td>
                    <td>0.73</td>
                    <td>0.82</td>
                    <td>0.77</td>
                    <td>55</td>
                  </tr>
                  <tr className="cal-row severe-row">
                    <td><span className="cal-dot" style={{ backgroundColor: '#881337' }}></span>Severe</td>
                    <td>0.67</td>
                    <td>0.43</td>
                    <td>0.52</td>
                    <td>14</td>
                  </tr>
                  <tr className="cal-row accuracy-row">
                    <td><strong>Accuracy</strong></td>
                    <td colSpan="2"></td>
                    <td><strong>0.77</strong></td>
                    <td>180</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        )}

        {activeDiagnosticsTab === 'stats' && (
          <div className="stats-tab-content">
            {result && result.stats ? (
              <>
                <p className="tab-desc">Dynamic historical distributions calculated for <strong>{result.stats.month_name}</strong> in Delhi:</p>
                <div className="stats-metrics-grid">
                  <div className="stats-card">
                    <span className="stats-card-label">Mean (Average)</span>
                    <span className="stats-card-value">{result.stats.mean}</span>
                  </div>
                  <div className="stats-card">
                    <span className="stats-card-label">Median (Middle)</span>
                    <span className="stats-card-value">{result.stats.median}</span>
                  </div>
                  <div className="stats-card">
                    <span className="stats-card-label">Mode (Common)</span>
                    <span className="stats-card-value">{result.stats.mode}</span>
                  </div>
                  <div className="stats-card">
                    <span className="stats-card-label">Std Deviation</span>
                    <span className="stats-card-value">±{result.stats.std_dev}</span>
                  </div>
                  <div className="stats-card" style={{ gridColumn: 'span 2' }}>
                    <span className="stats-card-label">Variance (Spread)</span>
                    <span className="stats-card-value" style={{ fontSize: '1.15rem' }}>{result.stats.variance}</span>
                  </div>
                  <div className="stats-card" style={{ gridColumn: 'span 2', display: 'flex', flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', padding: '0.9rem 1.1rem' }}>
                    <div>
                      <span className="stats-card-label" style={{ marginBottom: '0.1rem' }}>Annual Linear Trend</span>
                      <span className="trend-desc" style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Multi-year slope of {result.stats.month_name} averages</span>
                    </div>
                    <div style={{ textAlign: 'right' }}>
                      <span className="stats-card-value" style={{ 
                        fontSize: '1.05rem', 
                        color: result.stats.trend_slope < 0 ? '#16a34a' : '#dc2626',
                        fontWeight: 800
                      }}>
                        {result.stats.trend_slope > 0 ? `+${result.stats.trend_slope}` : result.stats.trend_slope} AQI/yr
                      </span>
                      <span style={{ 
                        display: 'block', 
                        fontSize: '0.65rem', 
                        fontWeight: 700, 
                        color: result.stats.trend_slope < 0 ? '#16a34a' : '#dc2626'
                      }}>
                        {result.stats.trend_slope < 0 ? '🟢 Improving Trend' : '🔴 Worsening Trend'}
                      </span>
                    </div>
                  </div>
                </div>
              </>
            ) : (
              <div className="stats-locked">
                <span style={{ fontSize: '1.5rem', marginBottom: '0.5rem', display: 'block' }}>🔒</span>
                <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Awaiting prediction date to analyze seasonal stats.</p>
              </div>
            )}
          </div>
        )}

        {activeDiagnosticsTab === 'distribution' && (
          <div className="distribution-tab-content">
            {result && result.stats && result.stats.pdf && result.stats.pdf.length > 0 ? (
              <>
                <p className="tab-desc">Normal Probability Distribution PDF of Delhi AQI in <strong>{result.stats.month_name}</strong>:</p>
                {renderNormalCurve(result)}
                <div className="distribution-legend" style={{ marginTop: '0.75rem', display: 'flex', gap: '1rem', justifyContent: 'center' }}>
                  <span className="legend-item" style={{ display: 'inline-flex', alignItems: 'center', gap: '0.35rem', fontSize: '0.7rem', color: 'var(--text-muted)' }}>
                    <span className="legend-line" style={{ display: 'inline-block', width: '12px', height: '3px', backgroundColor: 'var(--primary)', borderRadius: '2px' }}></span>
                    Normal Curve
                  </span>
                  <span className="legend-item" style={{ display: 'inline-flex', alignItems: 'center', gap: '0.35rem', fontSize: '0.7rem', color: 'var(--text-muted)' }}>
                    <span className="legend-line" style={{ display: 'inline-block', width: '12px', height: '0px', borderTop: '2px dashed #be123c' }}></span>
                    Predicted AQI ({result.predicted_aqi})
                  </span>
                </div>
              </>
            ) : (
              <div className="stats-locked">
                <span style={{ fontSize: '1.5rem', marginBottom: '0.5rem', display: 'block' }}>🔒</span>
                <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Awaiting prediction date to plot probability curve.</p>
              </div>
            )}
          </div>
        )}
      </div>
    </div>

    {/* Right Panel */}
      <div className="right-panel">
        {loading && (
          <div className="card spinner-container">
            <div className="spinner"></div>
            <h3 style={{ fontWeight: 700 }}>Synthesizing Temporal Features...</h3>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>Running advanced regression algorithms</p>
          </div>
        )}

        {error && (
          <div className="error-card">
            <span style={{ fontSize: '1.5rem' }}>⚠️</span>
            <div>
              <strong style={{ display: 'block', marginBottom: '0.1rem' }}>Data Lookup Failed</strong>
              <span style={{ fontSize: '0.85rem', opacity: 0.9 }}>{error}</span>
            </div>
          </div>
        )}

        {!loading && !error && !result && (
          <div className="empty-state">
            <div className="empty-state-icon">🔮</div>
            <h3>Predictive Analysis Ready</h3>
            <p>Select any historical calendar date or simulate any future date in the panel to generate advanced air quality telemetry.</p>
          </div>
        )}

        {!loading && !error && result && (
          <>
            {/* Split result and weather details */}
            <div className="result-layout">
              {/* Prediction main visual */}
              <div 
                className="card result-main"
                style={{ 
                  '--aqi-color': result.color,
                  '--aqi-color-shadow': `${result.color}33`
                }}
              >
                <span className="aqi-pill">{result.category}</span>
                <div className="metric-gauge">
                  <div className="metric-ring"></div>
                  <div style={{ textAlign: 'center' }}>
                    <div className="metric-aqi-val">{Math.round(result.predicted_aqi)}</div>
                    <div className="metric-aqi-label">Predict AQI</div>
                  </div>
                </div>
                {result.is_future ? (
                  <div style={{ marginTop: '0.5rem', color: '#4f46e5', fontSize: '0.75rem', fontWeight: 800, backgroundColor: '#f5f3ff', padding: '0.3rem 0.75rem', borderRadius: '9999px', border: '1px solid rgba(79, 70, 229, 0.15)', display: 'inline-block', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                    🔮 Climatology Simulation
                  </div>
                ) : (
                  result.actual_aqi && (
                    <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)', fontWeight: 600 }}>
                      Actual Recorded AQI: <span style={{ color: '#1e293b', fontWeight: 800 }}>{result.actual_aqi}</span>
                    </div>
                  )
                )}
              </div>

              {/* Weather info grids */}
              <div className="weather-grid">
                <div className="weather-card">
                  {getWeatherIcon('temp')}
                  <div className="w-details">
                    <span className="w-label">Temperature</span>
                    <span className="w-value">{result.weather.temp}°C</span>
                  </div>
                </div>
                <div className="weather-card">
                  {getWeatherIcon('humidity')}
                  <div className="w-details">
                    <span className="w-label">Humidity</span>
                    <span className="w-value">{result.weather.humidity}%</span>
                  </div>
                </div>
                <div className="weather-card">
                  {getWeatherIcon('wind')}
                  <div className="w-details">
                    <span className="w-label">Wind Speed</span>
                    <span className="w-value">{result.weather.windspeed} km/h</span>
                  </div>
                </div>
                <div className="weather-card">
                  {getWeatherIcon('precip')}
                  <div className="w-details">
                    <span className="w-label">Rainfall</span>
                    <span className="w-value">{result.weather.precip} mm</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Health & safety Advice section */}
            {advice && (
              <div className="card advice-block" style={{ '--aqi-color': result.color }}>
                <div className="advice-icon">{advice.icon}</div>
                <div className="advice-content">
                  <h4>{advice.title}</h4>
                  <p>{advice.text}</p>
                </div>
              </div>
            )}

            {/* Historical trend charts */}
            {result.trend && result.trend.length > 0 && (
              <div className="card trend-section">
                <div className="section-header">
                  <h3 className="section-title">Preceding 7-Day Context</h3>
                  <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)', fontWeight: 700 }}>Recorded Historical Data</span>
                </div>
                {renderTrendChart(result.trend)}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

export default App;
