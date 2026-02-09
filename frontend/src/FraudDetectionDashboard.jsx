import React, { useState, useEffect } from 'react';
import { BarChart, Bar, LineChart, Line, PieChart, Pie, Cell, RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ScatterChart, Scatter } from 'recharts';
import { AlertCircle, CheckCircle, TrendingUp, DollarSign, Shield, Activity, FileText, Search } from 'lucide-react';

const FraudDetectionDashboard = () => {
  const [activeTab, setActiveTab] = useState('overview');
  const [predictionForm, setPredictionForm] = useState({
    age: 35,
    claim_amount: 5000,
    policy_tenure_months: 24,
    vehicle_value: 25000,
    police_report_filed: 1,
    witness_count: 2,
    number_of_previous_claims: 0,
    claim_report_delay_hours: 2
  });
  const [predictionResult, setPredictionResult] = useState(null);
  const [loading, setLoading] = useState(false);

  // Sample data - in production, this would come from your Flask API
  const fraudDistribution = [
    { name: 'Legitimate', value: 12750, percentage: 85 },
    { name: 'Fraudulent', value: 2250, percentage: 15 }
  ];

  const modelPerformance = [
    { model: 'Logistic Regression', accuracy: 0.87, precision: 0.84, recall: 0.82, f1: 0.83, rocAuc: 0.89 },
    { model: 'Decision Tree', accuracy: 0.89, precision: 0.86, recall: 0.85, f1: 0.85, rocAuc: 0.91 },
    { model: 'Random Forest', accuracy: 0.93, precision: 0.91, recall: 0.90, f1: 0.90, rocAuc: 0.96 },
    { model: 'Gradient Boosting', accuracy: 0.94, precision: 0.92, recall: 0.91, f1: 0.91, rocAuc: 0.97 },
    { model: 'XGBoost', accuracy: 0.95, precision: 0.94, recall: 0.93, f1: 0.93, rocAuc: 0.98 },
    { model: 'LightGBM', accuracy: 0.95, precision: 0.93, recall: 0.92, f1: 0.92, rocAuc: 0.97 },
    { model: 'Neural Network', accuracy: 0.93, precision: 0.91, recall: 0.89, f1: 0.90, rocAuc: 0.95 }
  ];

  const fraudByIncident = [
    { type: 'Theft', rate: 28 },
    { type: 'Fire', rate: 25 },
    { type: 'Hit and Run', rate: 19 },
    { type: 'Vandalism', rate: 17 },
    { type: 'Natural Disaster', rate: 14 },
    { type: 'Collision', rate: 11 }
  ];

  const hourlyFraud = [
    { hour: '0', rate: 22 }, { hour: '3', rate: 25 }, { hour: '6', rate: 12 },
    { hour: '9', rate: 10 }, { hour: '12', rate: 13 }, { hour: '15', rate: 11 },
    { hour: '18', rate: 15 }, { hour: '21', rate: 19 }, { hour: '23', rate: 24 }
  ];

  const featureImportance = [
    { feature: 'claim_to_vehicle_ratio', importance: 0.156 },
    { feature: 'claim_amount', importance: 0.142 },
    { feature: 'evidence_score', importance: 0.128 },
    { feature: 'policy_tenure_months', importance: 0.115 },
    { feature: 'number_of_previous_claims', importance: 0.098 },
    { feature: 'claim_report_delay_hours', importance: 0.087 },
    { feature: 'witness_count', importance: 0.074 },
    { feature: 'vehicle_age', importance: 0.065 },
    { feature: 'high_value_claim', importance: 0.058 },
    { feature: 'new_policy_high_claim', importance: 0.077 }
  ];

  const radarData = [
    { metric: 'Accuracy', XGBoost: 0.95, RandomForest: 0.93, NeuralNet: 0.93 },
    { metric: 'Precision', XGBoost: 0.94, RandomForest: 0.91, NeuralNet: 0.91 },
    { metric: 'Recall', XGBoost: 0.93, RandomForest: 0.90, NeuralNet: 0.89 },
    { metric: 'F1-Score', XGBoost: 0.93, RandomForest: 0.90, NeuralNet: 0.90 },
    { metric: 'ROC-AUC', XGBoost: 0.98, RandomForest: 0.96, NeuralNet: 0.95 }
  ];

  const claimDistribution = [
    { range: '0-5K', legitimate: 3500, fraudulent: 200 },
    { range: '5-10K', legitimate: 2800, fraudulent: 350 },
    { range: '10-20K', legitimate: 2200, fraudulent: 580 },
    { range: '20-30K', legitimate: 1500, fraudulent: 420 },
    { range: '30K+', legitimate: 750, fraudulent: 700 }
  ];

  const COLORS = ['#10b981', '#ef4444', '#3b82f6', '#f59e0b', '#8b5cf6', '#ec4899'];

  const handlePrediction = async () => {
    setLoading(true);
    
    // Simulate API call
    setTimeout(() => {
      const score = calculateRiskScore(predictionForm);
      const probability = Math.min(score / 100, 0.95);
      const isValid = probability > 0.5;
      
      setPredictionResult({
        is_fraud: isValid,
        fraud_probability: probability * 100,
        risk_level: probability < 0.3 ? 'Low' : probability < 0.6 ? 'Medium' : probability < 0.8 ? 'High' : 'Critical',
        recommendation: probability < 0.3 ? 'Approve with standard processing' : 
                       probability < 0.6 ? 'Additional verification recommended' :
                       probability < 0.8 ? 'Thorough investigation required' :
                       'Immediate fraud investigation required',
        confidence: Math.abs(probability - 0.5) > 0.3 ? 'High' : Math.abs(probability - 0.5) > 0.15 ? 'Medium' : 'Low'
      });
      setLoading(false);
    }, 1000);
  };

  const calculateRiskScore = (form) => {
    let score = 0;
    if (form.claim_amount > 10000) score += 30;
    if (form.policy_tenure_months < 12) score += 25;
    if (form.number_of_previous_claims > 2) score += 20;
    if (form.witness_count === 0) score += 15;
    if (form.police_report_filed === 0) score += 10;
    if (form.claim_report_delay_hours > 24) score += 15;
    return score;
  };

  const StatCard = ({ icon: Icon, title, value, subtitle, color }) => (
    <div className="bg-white rounded-lg shadow-md p-6 border-l-4" style={{ borderLeftColor: color }}>
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm text-gray-600 font-medium">{title}</p>
          <p className="text-3xl font-bold mt-2" style={{ color }}>{value}</p>
          <p className="text-xs text-gray-500 mt-1">{subtitle}</p>
        </div>
        <div className="p-3 rounded-full" style={{ backgroundColor: `${color}20` }}>
          <Icon size={32} style={{ color }} />
        </div>
      </div>
    </div>
  );

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-gradient-to-r from-blue-600 to-purple-600 text-white shadow-lg">
        <div className="max-w-7xl mx-auto px-6 py-8">
          <div className="flex items-center gap-3">
            <Shield size={40} />
            <div>
              <h1 className="text-3xl font-bold">Insurance Fraud Detection System</h1>
              <p className="text-blue-100 mt-1">Advanced ML-Powered Analytics & Prediction Platform</p>
            </div>
          </div>
        </div>
      </div>

      {/* Navigation Tabs */}
      <div className="bg-white shadow-sm border-b sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-6">
          <div className="flex gap-1">
            {[
              { id: 'overview', label: 'Overview', icon: Activity },
              { id: 'models', label: 'Model Performance', icon: TrendingUp },
              { id: 'analysis', label: 'Data Analysis', icon: BarChart },
              { id: 'predict', label: 'Live Prediction', icon: Search }
            ].map(tab => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center gap-2 px-6 py-4 font-medium transition-colors ${
                  activeTab === tab.id
                    ? 'text-blue-600 border-b-2 border-blue-600'
                    : 'text-gray-600 hover:text-gray-900'
                }`}
              >
                <tab.icon size={18} />
                {tab.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-6 py-8">
        {activeTab === 'overview' && (
          <div className="space-y-8">
            {/* Key Metrics */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              <StatCard
                icon={FileText}
                title="Total Claims"
                value="15,000"
                subtitle="Generated for analysis"
                color="#3b82f6"
              />
              <StatCard
                icon={AlertCircle}
                title="Fraud Rate"
                value="15%"
                subtitle="2,250 fraudulent claims"
                color="#ef4444"
              />
              <StatCard
                icon={TrendingUp}
                title="Model Accuracy"
                value="95%"
                subtitle="XGBoost best performer"
                color="#10b981"
              />
              <StatCard
                icon={DollarSign}
                title="Net Savings"
                value="$13.5M"
                subtitle="After investigation costs"
                color="#f59e0b"
              />
            </div>

            {/* Charts Row 1 */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Fraud Distribution */}
              <div className="bg-white rounded-lg shadow-md p-6">
                <h3 className="text-lg font-bold text-gray-800 mb-4">Fraud Distribution</h3>
                <ResponsiveContainer width="100%" height={300}>
                  <PieChart>
                    <Pie
                      data={fraudDistribution}
                      cx="50%"
                      cy="50%"
                      labelLine={false}
                      label={({ name, percentage }) => `${name}: ${percentage}%`}
                      outerRadius={100}
                      fill="#8884d8"
                      dataKey="value"
                    >
                      {fraudDistribution.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={COLORS[index]} />
                      ))}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              </div>

              {/* Fraud Rate by Incident Type */}
              <div className="bg-white rounded-lg shadow-md p-6">
                <h3 className="text-lg font-bold text-gray-800 mb-4">Fraud Rate by Incident Type</h3>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={fraudByIncident} layout="vertical">
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis type="number" />
                    <YAxis dataKey="type" type="category" width={100} />
                    <Tooltip />
                    <Bar dataKey="rate" fill="#3b82f6" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Charts Row 2 */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Hourly Fraud Pattern */}
              <div className="bg-white rounded-lg shadow-md p-6">
                <h3 className="text-lg font-bold text-gray-800 mb-4">Fraud Rate by Hour of Day</h3>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={hourlyFraud}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="hour" />
                    <YAxis />
                    <Tooltip />
                    <Line type="monotone" dataKey="rate" stroke="#f59e0b" strokeWidth={3} />
                  </LineChart>
                </ResponsiveContainer>
              </div>

              {/* Claim Distribution */}
              <div className="bg-white rounded-lg shadow-md p-6">
                <h3 className="text-lg font-bold text-gray-800 mb-4">Claim Amount Distribution</h3>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={claimDistribution}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="range" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Bar dataKey="legitimate" fill="#10b981" name="Legitimate" />
                    <Bar dataKey="fraudulent" fill="#ef4444" name="Fraudulent" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Financial Impact */}
            <div className="bg-gradient-to-r from-green-50 to-blue-50 rounded-lg shadow-md p-6 border border-green-200">
              <h3 className="text-lg font-bold text-gray-800 mb-4">💰 Financial Impact Analysis</h3>
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <div className="bg-white rounded-lg p-4 shadow">
                  <p className="text-sm text-gray-600">Total Claims</p>
                  <p className="text-2xl font-bold text-gray-900">$187.5M</p>
                </div>
                <div className="bg-white rounded-lg p-4 shadow">
                  <p className="text-sm text-gray-600">Fraud Detected</p>
                  <p className="text-2xl font-bold text-red-600">$28.1M</p>
                </div>
                <div className="bg-white rounded-lg p-4 shadow">
                  <p className="text-sm text-gray-600">Investigation Cost</p>
                  <p className="text-2xl font-bold text-orange-600">$1.1M</p>
                </div>
                <div className="bg-white rounded-lg p-4 shadow">
                  <p className="text-sm text-gray-600">Net Savings</p>
                  <p className="text-2xl font-bold text-green-600">$13.5M</p>
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'models' && (
          <div className="space-y-8">
            {/* Model Comparison Table */}
            <div className="bg-white rounded-lg shadow-md p-6">
              <h3 className="text-lg font-bold text-gray-800 mb-4">Model Performance Comparison</h3>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-4 py-3 text-left font-semibold text-gray-700">Model</th>
                      <th className="px-4 py-3 text-center font-semibold text-gray-700">Accuracy</th>
                      <th className="px-4 py-3 text-center font-semibold text-gray-700">Precision</th>
                      <th className="px-4 py-3 text-center font-semibold text-gray-700">Recall</th>
                      <th className="px-4 py-3 text-center font-semibold text-gray-700">F1-Score</th>
                      <th className="px-4 py-3 text-center font-semibold text-gray-700">ROC-AUC</th>
                    </tr>
                  </thead>
                  <tbody>
                    {modelPerformance.map((model, idx) => (
                      <tr key={idx} className={`border-b ${model.model === 'XGBoost' ? 'bg-green-50' : ''}`}>
                        <td className="px-4 py-3 font-medium text-gray-900">
                          {model.model}
                          {model.model === 'XGBoost' && <span className="ml-2 text-xs bg-green-500 text-white px-2 py-1 rounded">BEST</span>}
                        </td>
                        <td className="px-4 py-3 text-center">{(model.accuracy * 100).toFixed(1)}%</td>
                        <td className="px-4 py-3 text-center">{(model.precision * 100).toFixed(1)}%</td>
                        <td className="px-4 py-3 text-center">{(model.recall * 100).toFixed(1)}%</td>
                        <td className="px-4 py-3 text-center">{(model.f1 * 100).toFixed(1)}%</td>
                        <td className="px-4 py-3 text-center font-semibold text-blue-600">{(model.rocAuc * 100).toFixed(1)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Radar Chart */}
            <div className="bg-white rounded-lg shadow-md p-6">
              <h3 className="text-lg font-bold text-gray-800 mb-4">Top 3 Models - Performance Radar</h3>
              <ResponsiveContainer width="100%" height={400}>
                <RadarChart data={radarData}>
                  <PolarGrid />
                  <PolarAngleAxis dataKey="metric" />
                  <PolarRadiusAxis angle={90} domain={[0, 1]} />
                  <Radar name="XGBoost" dataKey="XGBoost" stroke="#10b981" fill="#10b981" fillOpacity={0.5} />
                  <Radar name="Random Forest" dataKey="RandomForest" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.3} />
                  <Radar name="Neural Network" dataKey="NeuralNet" stroke="#8b5cf6" fill="#8b5cf6" fillOpacity={0.3} />
                  <Legend />
                  <Tooltip />
                </RadarChart>
              </ResponsiveContainer>
            </div>

            {/* Model Metrics Visualization */}
            <div className="bg-white rounded-lg shadow-md p-6">
              <h3 className="text-lg font-bold text-gray-800 mb-4">Detailed Metrics Comparison</h3>
              <ResponsiveContainer width="100%" height={400}>
                <BarChart data={modelPerformance}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="model" angle={-45} textAnchor="end" height={100} />
                  <YAxis domain={[0.75, 1]} />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="accuracy" fill="#3b82f6" name="Accuracy" />
                  <Bar dataKey="precision" fill="#10b981" name="Precision" />
                  <Bar dataKey="recall" fill="#f59e0b" name="Recall" />
                  <Bar dataKey="rocAuc" fill="#8b5cf6" name="ROC-AUC" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {activeTab === 'analysis' && (
          <div className="space-y-8">
            {/* Feature Importance */}
            <div className="bg-white rounded-lg shadow-md p-6">
              <h3 className="text-lg font-bold text-gray-800 mb-4">Top 10 Feature Importance</h3>
              <ResponsiveContainer width="100%" height={400}>
                <BarChart data={featureImportance} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" />
                  <YAxis dataKey="feature" type="category" width={200} />
                  <Tooltip />
                  <Bar dataKey="importance" fill="#8b5cf6">
                    {featureImportance.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Key Insights */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="bg-white rounded-lg shadow-md p-6">
                <h3 className="text-lg font-bold text-gray-800 mb-4">🔍 Key Risk Indicators</h3>
                <div className="space-y-4">
                  <div className="flex justify-between items-center p-3 bg-red-50 rounded">
                    <span className="font-medium text-gray-700">High claim-to-vehicle ratio</span>
                    <span className="text-red-600 font-bold">15.6%</span>
                  </div>
                  <div className="flex justify-between items-center p-3 bg-orange-50 rounded">
                    <span className="font-medium text-gray-700">Large claim amount</span>
                    <span className="text-orange-600 font-bold">14.2%</span>
                  </div>
                  <div className="flex justify-between items-center p-3 bg-yellow-50 rounded">
                    <span className="font-medium text-gray-700">Low evidence score</span>
                    <span className="text-yellow-600 font-bold">12.8%</span>
                  </div>
                  <div className="flex justify-between items-center p-3 bg-blue-50 rounded">
                    <span className="font-medium text-gray-700">Short policy tenure</span>
                    <span className="text-blue-600 font-bold">11.5%</span>
                  </div>
                  <div className="flex justify-between items-center p-3 bg-purple-50 rounded">
                    <span className="font-medium text-gray-700">Multiple previous claims</span>
                    <span className="text-purple-600 font-bold">9.8%</span>
                  </div>
                </div>
              </div>

              <div className="bg-white rounded-lg shadow-md p-6">
                <h3 className="text-lg font-bold text-gray-800 mb-4">📊 Dataset Statistics</h3>
                <div className="space-y-4">
                  <div className="border-b pb-3">
                    <p className="text-sm text-gray-600">Total Samples</p>
                    <p className="text-2xl font-bold text-gray-900">15,000</p>
                  </div>
                  <div className="border-b pb-3">
                    <p className="text-sm text-gray-600">Total Features</p>
                    <p className="text-2xl font-bold text-blue-600">58</p>
                  </div>
                  <div className="border-b pb-3">
                    <p className="text-sm text-gray-600">Engineered Features</p>
                    <p className="text-2xl font-bold text-purple-600">29</p>
                  </div>
                  <div className="pb-3">
                    <p className="text-sm text-gray-600">Training/Test Split</p>
                    <p className="text-2xl font-bold text-green-600">80% / 20%</p>
                  </div>
                </div>
              </div>
            </div>

            {/* Evidence Impact */}
            <div className="bg-white rounded-lg shadow-md p-6">
              <h3 className="text-lg font-bold text-gray-800 mb-4">Impact of Evidence on Fraud Rate</h3>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={[
                  { evidence: 'Police Report', with: 12, without: 23 },
                  { evidence: 'Witnesses', with: 10, without: 22 },
                  { evidence: 'Photos', with: 11, without: 21 }
                ]}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="evidence" />
                  <YAxis label={{ value: 'Fraud Rate (%)', angle: -90, position: 'insideLeft' }} />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="with" fill="#10b981" name="With Evidence" />
                  <Bar dataKey="without" fill="#ef4444" name="Without Evidence" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {activeTab === 'predict' && (
          <div className="space-y-8">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              {/* Prediction Form */}
              <div className="bg-white rounded-lg shadow-md p-6">
                <h3 className="text-lg font-bold text-gray-800 mb-6">📝 Enter Claim Details</h3>
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Age</label>
                    <input
                      type="number"
                      value={predictionForm.age}
                      onChange={(e) => setPredictionForm({...predictionForm, age: parseInt(e.target.value)})}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    />
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Claim Amount ($)</label>
                    <input
                      type="number"
                      value={predictionForm.claim_amount}
                      onChange={(e) => setPredictionForm({...predictionForm, claim_amount: parseInt(e.target.value)})}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Policy Tenure (months)</label>
                    <input
                      type="number"
                      value={predictionForm.policy_tenure_months}
                      onChange={(e) => setPredictionForm({...predictionForm, policy_tenure_months: parseInt(e.target.value)})}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Vehicle Value ($)</label>
                    <input
                      type="number"
                      value={predictionForm.vehicle_value}
                      onChange={(e) => setPredictionForm({...predictionForm, vehicle_value: parseInt(e.target.value)})}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Police Report Filed</label>
                    <select
                      value={predictionForm.police_report_filed}
                      onChange={(e) => setPredictionForm({...predictionForm, police_report_filed: parseInt(e.target.value)})}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    >
                      <option value={1}>Yes</option>
                      <option value={0}>No</option>
                    </select>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Number of Witnesses</label>
                    <input
                      type="number"
                      value={predictionForm.witness_count}
                      onChange={(e) => setPredictionForm({...predictionForm, witness_count: parseInt(e.target.value)})}
                      min="0"
                      max="10"
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Previous Claims</label>
                    <input
                      type="number"
                      value={predictionForm.number_of_previous_claims}
                      onChange={(e) => setPredictionForm({...predictionForm, number_of_previous_claims: parseInt(e.target.value)})}
                      min="0"
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Claim Report Delay (hours)</label>
                    <input
                      type="number"
                      value={predictionForm.claim_report_delay_hours}
                      onChange={(e) => setPredictionForm({...predictionForm, claim_report_delay_hours: parseInt(e.target.value)})}
                      min="0"
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    />
                  </div>

                  <button
                    onClick={handlePrediction}
                    disabled={loading}
                    className="w-full bg-gradient-to-r from-blue-600 to-purple-600 text-white py-3 rounded-md font-semibold hover:from-blue-700 hover:to-purple-700 transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                  >
                    {loading ? (
                      <>
                        <div className="animate-spin h-5 w-5 border-2 border-white border-t-transparent rounded-full"></div>
                        Processing...
                      </>
                    ) : (
                      <>
                        <Search size={20} />
                        Analyze Claim
                      </>
                    )}
                  </button>
                </div>
              </div>

              {/* Prediction Result */}
              <div className="bg-white rounded-lg shadow-md p-6">
                <h3 className="text-lg font-bold text-gray-800 mb-6">🎯 Prediction Results</h3>
                
                {!predictionResult ? (
                  <div className="flex flex-col items-center justify-center h-full text-gray-400 py-12">
                    <Search size={64} className="mb-4" />
                    <p className="text-lg">Enter claim details and click "Analyze Claim"</p>
                    <p className="text-sm mt-2">Get instant fraud probability assessment</p>
                  </div>
                ) : (
                  <div className="space-y-6">
                    {/* Fraud Status */}
                    <div className={`p-6 rounded-lg border-2 ${
                      predictionResult.is_fraud 
                        ? 'bg-red-50 border-red-300' 
                        : 'bg-green-50 border-green-300'
                    }`}>
                      <div className="flex items-center gap-3 mb-2">
                        {predictionResult.is_fraud ? (
                          <AlertCircle size={32} className="text-red-600" />
                        ) : (
                          <CheckCircle size={32} className="text-green-600" />
                        )}
                        <div>
                          <p className="text-sm text-gray-600">Fraud Detection</p>
                          <p className={`text-2xl font-bold ${
                            predictionResult.is_fraud ? 'text-red-600' : 'text-green-600'
                          }`}>
                            {predictionResult.is_fraud ? '⚠️ FRAUDULENT' : '✅ LEGITIMATE'}
                          </p>
                        </div>
                      </div>
                    </div>

                    {/* Probability */}
                    <div className="bg-gray-50 p-6 rounded-lg">
                      <p className="text-sm text-gray-600 mb-2">Fraud Probability</p>
                      <div className="flex items-end gap-2">
                        <p className="text-4xl font-bold text-blue-600">
                          {predictionResult.fraud_probability.toFixed(2)}%
                        </p>
                        <p className="text-sm text-gray-500 mb-2">confidence: {predictionResult.confidence}</p>
                      </div>
                      <div className="mt-3 bg-gray-200 rounded-full h-3 overflow-hidden">
                        <div 
                          className={`h-full transition-all duration-500 ${
                            predictionResult.fraud_probability < 30 ? 'bg-green-500' :
                            predictionResult.fraud_probability < 60 ? 'bg-yellow-500' :
                            predictionResult.fraud_probability < 80 ? 'bg-orange-500' :
                            'bg-red-500'
                          }`}
                          style={{ width: `${predictionResult.fraud_probability}%` }}
                        ></div>
                      </div>
                    </div>

                    {/* Risk Level */}
                    <div className="grid grid-cols-2 gap-4">
                      <div className={`p-4 rounded-lg ${
                        predictionResult.risk_level === 'Critical' ? 'bg-red-100 border border-red-300' :
                        predictionResult.risk_level === 'High' ? 'bg-orange-100 border border-orange-300' :
                        predictionResult.risk_level === 'Medium' ? 'bg-yellow-100 border border-yellow-300' :
                        'bg-green-100 border border-green-300'
                      }`}>
                        <p className="text-sm text-gray-600 mb-1">Risk Level</p>
                        <p className={`text-xl font-bold ${
                          predictionResult.risk_level === 'Critical' ? 'text-red-700' :
                          predictionResult.risk_level === 'High' ? 'text-orange-700' :
                          predictionResult.risk_level === 'Medium' ? 'text-yellow-700' :
                          'text-green-700'
                        }`}>
                          {predictionResult.risk_level}
                        </p>
                      </div>

                      <div className="p-4 rounded-lg bg-blue-100 border border-blue-300">
                        <p className="text-sm text-gray-600 mb-1">Confidence</p>
                        <p className="text-xl font-bold text-blue-700">{predictionResult.confidence}</p>
                      </div>
                    </div>

                    {/* Recommendation */}
                    <div className="bg-gradient-to-r from-purple-50 to-blue-50 p-6 rounded-lg border border-purple-200">
                      <p className="text-sm text-gray-600 mb-2">💡 Recommendation</p>
                      <p className="text-base font-semibold text-gray-800">
                        {predictionResult.recommendation}
                      </p>
                    </div>

                    {/* Action Button */}
                    <button
                      onClick={() => setPredictionResult(null)}
                      className="w-full bg-gray-600 text-white py-2 rounded-md font-medium hover:bg-gray-700 transition-colors"
                    >
                      Clear & Analyze Another Claim
                    </button>
                  </div>
                )}
              </div>
            </div>

            {/* Example Claims */}
            <div className="bg-white rounded-lg shadow-md p-6">
              <h3 className="text-lg font-bold text-gray-800 mb-4">📋 Quick Test Examples</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <button
                  onClick={() => setPredictionForm({
                    age: 24,
                    claim_amount: 25000,
                    policy_tenure_months: 2,
                    vehicle_value: 20000,
                    police_report_filed: 0,
                    witness_count: 0,
                    number_of_previous_claims: 4,
                    claim_report_delay_hours: 72
                  })}
                  className="p-4 border-2 border-red-200 rounded-lg hover:bg-red-50 transition-colors text-left"
                >
                  <p className="font-semibold text-red-600 mb-2">🔴 High-Risk Claim</p>
                  <p className="text-sm text-gray-600">New policy, large claim, no evidence</p>
                </button>

                <button
                  onClick={() => setPredictionForm({
                    age: 45,
                    claim_amount: 3500,
                    policy_tenure_months: 48,
                    vehicle_value: 28000,
                    police_report_filed: 1,
                    witness_count: 2,
                    number_of_previous_claims: 0,
                    claim_report_delay_hours: 2
                  })}
                  className="p-4 border-2 border-green-200 rounded-lg hover:bg-green-50 transition-colors text-left"
                >
                  <p className="font-semibold text-green-600 mb-2">🟢 Low-Risk Claim</p>
                  <p className="text-sm text-gray-600">Long-term policy, documented evidence</p>
                </button>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Footer */}
      <div className="bg-gray-800 text-white mt-16">
        <div className="max-w-7xl mx-auto px-6 py-8">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <div>
              <h4 className="font-bold mb-3">About Project</h4>
              <p className="text-gray-400 text-sm">
                Advanced machine learning system for detecting insurance fraud using 
                XGBoost with 95% accuracy and $13.5M net savings.
              </p>
            </div>
            <div>
              <h4 className="font-bold mb-3">Technologies</h4>
              <div className="flex flex-wrap gap-2">
                <span className="bg-gray-700 px-3 py-1 rounded text-xs">Python</span>
                <span className="bg-gray-700 px-3 py-1 rounded text-xs">XGBoost</span>
                <span className="bg-gray-700 px-3 py-1 rounded text-xs">Flask</span>
                <span className="bg-gray-700 px-3 py-1 rounded text-xs">React</span>
                <span className="bg-gray-700 px-3 py-1 rounded text-xs">SMOTE</span>
              </div>
            </div>
            <div>
              <h4 className="font-bold mb-3">Key Features</h4>
              <ul className="text-gray-400 text-sm space-y-1">
                <li>✓ 58 engineered features</li>
                <li>✓ 7 ML model comparison</li>
                <li>✓ Real-time predictions</li>
                <li>✓ Cost-benefit analysis</li>
              </ul>
            </div>
          </div>
          <div className="border-t border-gray-700 mt-8 pt-6 text-center text-gray-400 text-sm">
            <p>Insurance Fraud Detection System © 2024 | Built with Machine Learning & Advanced Analytics</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default FraudDetectionDashboard;