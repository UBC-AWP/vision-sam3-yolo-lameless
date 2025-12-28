import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import { AuthProvider } from './contexts/AuthContext'
import { ProtectedRoute, ResearcherRoute, AdminRoute } from './components/ProtectedRoute'
import Layout from './components/Layout'
import Dashboard from './pages/Dashboard'
import VideoUpload from './pages/VideoUpload'
import VideoAnalysis from './pages/VideoAnalysis'
import TrainingQueue from './pages/TrainingQueue'
import ModelConfig from './pages/ModelConfig'
import PairwiseReview from './pages/PairwiseReview'
import TripletComparison from './pages/TripletComparison'
import HierarchyVisualization from './pages/HierarchyVisualization'
import SimilarityMap from './pages/SimilarityMap'
import TrainingModule from './pages/TrainingModule'
import Login from './pages/Login'
import PipelineMonitor from './pages/PipelineMonitor'
import SystemHealth from './pages/SystemHealth'
import VideoResults from './pages/VideoResults'
import MLConfiguration from './pages/MLConfiguration'

function App() {
  return (
    <AuthProvider>
      <Router>
        <Routes>
          {/* Public route */}
          <Route path="/login" element={<Login />} />

          {/* Protected routes wrapped in Layout */}
          <Route
            path="/*"
            element={
              <Layout>
                <Routes>
                  {/* Dashboard - accessible to all authenticated users */}
                  <Route path="/" element={<Dashboard />} />

                  {/* Video management - researcher and above */}
                  <Route
                    path="/upload"
                    element={
                      <ResearcherRoute>
                        <VideoUpload />
                      </ResearcherRoute>
                    }
                  />
                  <Route path="/video/:videoId" element={<VideoAnalysis />} />
                  <Route path="/analysis/:videoId" element={<VideoAnalysis />} />
                  <Route path="/results/:videoId" element={<VideoResults />} />

                  {/* Human-in-the-loop - all authenticated users */}
                  <Route path="/pairwise" element={<PairwiseReview />} />
                  <Route path="/triplet" element={<TripletComparison />} />
                  <Route path="/compare/:videoId1/:videoId2" element={<PairwiseReview />} />

                  {/* Analytics - all authenticated users */}
                  <Route path="/hierarchy" element={<HierarchyVisualization />} />
                  <Route path="/similarity" element={<SimilarityMap />} />
                  <Route path="/learn" element={<TrainingModule />} />

                  {/* Training & Models - researcher and above */}
                  <Route
                    path="/training"
                    element={
                      <ResearcherRoute>
                        <TrainingQueue />
                      </ResearcherRoute>
                    }
                  />
                  <Route
                    path="/models"
                    element={
                      <ResearcherRoute>
                        <ModelConfig />
                      </ResearcherRoute>
                    }
                  />
                  <Route
                    path="/ml-config"
                    element={
                      <ResearcherRoute>
                        <MLConfiguration />
                      </ResearcherRoute>
                    }
                  />

                  {/* Pipeline & System - researcher and above */}
                  <Route
                    path="/pipelines"
                    element={
                      <ResearcherRoute>
                        <PipelineMonitor />
                      </ResearcherRoute>
                    }
                  />
                  <Route
                    path="/health"
                    element={
                      <ResearcherRoute>
                        <SystemHealth />
                      </ResearcherRoute>
                    }
                  />
                </Routes>
              </Layout>
            }
          />
        </Routes>
      </Router>
    </AuthProvider>
  )
}

export default App
