#include "init/gelib.h"

namespace ge {
    //Get Singleton Instance
    std::shared_ptr<GELib> GELib::GetInstance()
    {
        return nullptr;
    }

    bool OpsKernelManager::GetEnablePluginFlag() const {
        return true;
    }
}
