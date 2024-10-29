#include <QCoreApplication>
#include <QCommandLineParser>

#include "qfacefusion_api.h"

class CommandLineParser
{
private:
    CommandLineParser(void){
        static CommandLineParser::GC gc;
        parser.setApplicationDescription(
            "QFacefusion example, "
            "The following parameters can be configured to start the application:");
        parser.addHelpOption();
        parser.addVersionOption();
        foreach(QString opt,commandLineMap.keys()) {
            parser.addOption(
                commandLineMap.value(opt,QCommandLineOption("defaultValue")));
        }
    }
    static CommandLineParser* self;
    QCommandLineParser parser;
    bool processApp = false;
    QMap<QString, QCommandLineOption> commandLineMap = {
        {"path",
            QCommandLineOption(
                {"p","path"},
                "set model path",
                "mode path",
                ""
            )
        }
    };

public:
    void process(const QCoreApplication &app) {
        parser.process(app);
        processApp = true;
    }
    QString getOpt(const QString &optKey) const {
        if(processApp) {
            foreach(QString opt,commandLineMap.keys()) {
                if(opt == optKey){
                    QCommandLineOption dstOpt =
                        commandLineMap.value(opt,QCommandLineOption("defaultValue"));
                    if(parser.isSet(dstOpt)) {
                        return parser.value(dstOpt);
                    } else {
                        if(dstOpt.defaultValues().size() > 0)
                            return dstOpt.defaultValues().at(0);
                        else 
                            return "";
                    }
                }
            }
        }
        return "";
    }
    bool isSetOpt(const QString &optKey) const {
        if(processApp) {
            foreach(QString opt,commandLineMap.keys()) {
                if(opt == optKey){
                    QCommandLineOption dstOpt =
                        commandLineMap.value(opt,QCommandLineOption("defaultValue"));
                    return parser.isSet(dstOpt);
                }
            }
        }
        return false;
    }
    static CommandLineParser *instance() {
        if(!self) {
            self = new CommandLineParser();
        }
        return self;
    }

private:
    class GC
    {
    public:
        ~GC() {
            if (self != nullptr) {
                delete self;
                self = nullptr;
            }
        }
    };
};

CommandLineParser* CommandLineParser::self = nullptr;
#define  AppComLineParser   CommandLineParser::instance()

int main(int argc, char *argv[])
{
    QCoreApplication application(argc, argv);

    QCoreApplication::setApplicationName("QFacefusion example");
    QCoreApplication::setApplicationVersion("V1.0.0");

    AppComLineParser->process(application);

    QString path;
    if(AppComLineParser->isSetOpt("path")){
        path = AppComLineParser->getOpt("path");
    }

    if(path.isEmpty()) {
        qDebug() << "Please provide model paths";
        return -1;
    } else {
        qDebug() << "Model path: " << path;
    }

    std::string info = FaceFusion::getModelInfo(path.toStdString());
    qDebug("%s",info.c_str());

    return 0;
}
