import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import { AuthProvider } from './contexts/AuthContext'
import { ThemeProvider } from './contexts/ThemeContext'
import { LanguageProvider } from './contexts/LanguageContext'
import { ProtectedRoute, ResearcherRoute, AdminRoute } from './components/ProtectedRoute'
import Layout from './components/Layout'
import HomeRedirect from './pages/HomeRedirect'
import Dashboard from './pages/Dashboard'
import VideoUpload from './pages/VideoUpload'
import VideoAnalysis from './pages/VideoAnalysis'
import TrainingQueue from './pages/TrainingQueue'
import PairwiseReview from './pages/PairwiseReview'
import TripletComparison from './pages/TripletComparison'
import HierarchyVisualization from './pages/HierarchyVisualization'
import SimilarityMap from './pages/SimilarityMap'
import TrainingModule from './pages/TrainingModule'
import Performance from './pages/Performance'
import PairwiseResults from './pages/PairwiseResults'
import PairwiseTutorial from './pages/PairwiseTutorial'
import Login from './pages/Login'
import PipelineMonitor from './pages/PipelineMonitor'
import SystemHealth from './pages/SystemHealth'
import VideoResults from './pages/VideoResults'
import PipelineAnalysis from './pages/PipelineAnalysis'
import MLConfiguration from './pages/MLConfiguration'
import UserManagement from './pages/UserManagement'
import TutorialManagement from './pages/TutorialManagement'
import Settings from './pages/Settings'
import CowList from './pages/CowList'
import CowDetail from './pages/CowDetail'

function App() {
  return (
    <ThemeProvider>
      <LanguageProvider>
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
                    {/* Home route - role-aware landing */}
                    <Route path="/" element={<HomeRedirect />} />
                    <Route path="/dashboard" element={<Dashboard />} />

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
                    <Route
                      path="/pipeline-analysis/:videoId"
                      element={
                        <ResearcherRoute>
                          <PipelineAnalysis />
                        </ResearcherRoute>
                      }
                    />

                    {/* Human-in-the-loop - all authenticated users */}
                    <Route path="/pairwise" element={<PairwiseReview />} />
                    <Route path="/pairwise-tutorial" element={<PairwiseTutorial />} />
                    <Route path="/triplet" element={<TripletComparison />} />
                    <Route path="/compare/:videoId1/:videoId2" element={<PairwiseReview />} />

                    {/* Analytics - all authenticated users */}
                    <Route path="/hierarchy" element={<HierarchyVisualization />} />
                    <Route path="/similarity" element={<SimilarityMap />} />
                    <Route path="/performance" element={<Performance />} />
                    <Route path="/pair-results" element={<PairwiseResults />} />
                    <Route path="/learn" element={<TrainingModule />} />

                    {/* Cow Registry - all authenticated users */}
                    <Route path="/cows" element={<CowList />} />
                    <Route path="/cows/:cowId" element={<CowDetail />} />

                    {/* Settings - all authenticated users */}
                    <Route path="/settings" element={<Settings />} />

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
                      path="/config"
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

                    {/* Admin only routes */}
                    <Route
                      path="/users"
                      element={
                        <AdminRoute>
                          <UserManagement />
                        </AdminRoute>
                      }
                    />
                    <Route
                      path="/tutorials"
                      element={
                        <AdminRoute>
                          <TutorialManagement />
                        </AdminRoute>
                      }
                    />
                  </Routes>
                </Layout>
              }
            />
          </Routes>
          </Router>
        </AuthProvider>
      </LanguageProvider>
    </ThemeProvider>
  )
}

export default App
