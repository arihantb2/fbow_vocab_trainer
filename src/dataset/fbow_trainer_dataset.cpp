#include "dataset/fbow_trainer_dataset.h"

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <string>

namespace
{

std::string toLower(const std::string& s)
{
    std::string out = s;
    std::transform(out.begin(), out.end(), out.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return out;
}

std::vector<std::string> normalizeExtensions(const std::vector<std::string>& in)
{
    std::vector<std::string> out;
    out.reserve(in.size());
    for (size_t i = 0; i < in.size(); ++i)
    {
        std::string ext = toLower(in[i]);
        if (!ext.empty() && ext[0] != '.')
        {
            ext = "." + ext;
        }
        out.push_back(ext);
    }
    return out;
}

}  // namespace

std::vector<boost::filesystem::path> collectImages(const DatasetConfig& cfg)
{
    const std::vector<std::string> allowed = normalizeExtensions(cfg.extensions);

    std::vector<boost::filesystem::path> images;
    for (size_t d = 0; d < cfg.imagesDirs.size(); ++d)
    {
        const boost::filesystem::path root(cfg.imagesDirs[d]);
        if (!boost::filesystem::exists(root))
        {
            throw std::runtime_error("images directory does not exist: " + cfg.imagesDirs[d]);
        }
        if (!boost::filesystem::is_directory(root))
        {
            throw std::runtime_error("images path is not a directory: " + cfg.imagesDirs[d]);
        }

        if (cfg.recursive)
        {
            for (boost::filesystem::recursive_directory_iterator it(root), end; it != end; ++it)
            {
                if (!boost::filesystem::is_regular_file(*it))
                {
                    continue;
                }
                const std::string ext = toLower(it->path().extension().string());
                if (std::find(allowed.begin(), allowed.end(), ext) != allowed.end())
                {
                    images.push_back(it->path());
                }
            }
        }
        else
        {
            for (boost::filesystem::directory_iterator it(root), end; it != end; ++it)
            {
                if (!boost::filesystem::is_regular_file(*it))
                {
                    continue;
                }
                const std::string ext = toLower(it->path().extension().string());
                if (std::find(allowed.begin(), allowed.end(), ext) != allowed.end())
                {
                    images.push_back(it->path());
                }
            }
        }
    }

    std::sort(images.begin(), images.end());
    // Remove duplicates that may arise if directories overlap.
    images.erase(std::unique(images.begin(), images.end()), images.end());

    if (cfg.maxImages > 0 && static_cast<int>(images.size()) > cfg.maxImages)
    {
        images.resize(static_cast<size_t>(cfg.maxImages));
    }
    return images;
}

void ensureParentDirectory(const std::string& filePath)
{
    const boost::filesystem::path p(filePath);
    const boost::filesystem::path parent = p.parent_path();
    if (!parent.empty() && !boost::filesystem::exists(parent))
    {
        boost::filesystem::create_directories(parent);
    }
}
