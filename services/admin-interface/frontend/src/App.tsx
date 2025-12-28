import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
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

function App() {
  return (
    <Router>
      <Layout>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/upload" element={<VideoUpload />} />
          <Route path="/video/:videoId" element={<VideoAnalysis />} />
          <Route path="/analysis/:videoId" element={<VideoAnalysis />} />
          <Route path="/pairwise" element={<PairwiseReview />} />
          <Route path="/triplet" element={<TripletComparison />} />
          <Route path="/compare/:videoId1/:videoId2" element={<PairwiseReview />} />
          <Route path="/hierarchy" element={<HierarchyVisualization />} />
          <Route path="/similarity" element={<SimilarityMap />} />
          <Route path="/learn" element={<TrainingModule />} />
          <Route path="/training" element={<TrainingQueue />} />
          <Route path="/models" element={<ModelConfig />} />
        </Routes>
      </Layout>
    </Router>
  )
}

export default App
